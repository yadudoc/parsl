||| enough of pickle to make the interchange work, no more...
|||
||| makes heavy reference to Lib/pickletools.py (from around Python 3.12)
module Pickle

import Generics.Derive

import Bytes
import Logging

%language ElabReflection

-- It should be possible for this to be total, on the length of the
-- block of bytes, but this doesn't work at time of writing: I guess
-- I haven't conveyed that the recursion terminates properly.
-- %default total

||| some untyped (or Python Any-typed) representation of the output
||| of executing a pickle
public export
data PickleAST = PickleUnicodeString String
               | PickleInteger Int
               | PickleTuple (List PickleAST)
%runElab derive "PickleAST" [Generic, Meta, Show]

record VMState where
  constructor MkVMState
  stack: List PickleAST
  memo: List PickleAST  -- sequential map of integers to objects - memo key n is at position n in the list so we don't need to store the key


-- read an 8-byte integer - uint8 is the pickletools ArgumentDescriptor name.
-- if there are not 8 bytes left in ByteBlock, explode...
-- TODO: this could look nicer with a parser monad/parser applicative
-- rather than threading the shrinking bb?
read_uint8 : {n: Nat} -> ByteBlock (S (S (S (S (S (S (S (S n)))))))) -> IO (Int, ByteBlock n)
read_uint8 bb = do
  (byte1, bb1) <- bb_uncons bb 
  (byte2, bb2) <- bb_uncons bb1 
  (byte3, bb3) <- bb_uncons bb2
  (byte4, bb4) <- bb_uncons bb3 
  (byte5, bb5) <- bb_uncons bb4 
  (byte6, bb6) <- bb_uncons bb5 
  (byte7, bb7) <- bb_uncons bb6 
  (byte8, bb8) <- bb_uncons bb7 

  -- bytes are bytes but we want an Int...

  let v = (((((((
              byte8) * 256 + 
              byte7) * 256 + 
              byte6) * 256 + 
              byte5) * 256 + 
              byte4) * 256 + 
              byte3) * 256 + 
              byte2) * 256 +
              byte1
  pure (cast v, bb8)


-- define this signature before the body because we are mutually
-- recursive (or I could use a mutual block? what's the difference?)
step : {n : Nat} -> ByteBlock n -> VMState -> IO VMState

step_PROTO : {n : Nat} -> ByteBlock n -> VMState -> IO VMState

step_PROTO {n = Z} bb state = do
  ?error_PROTO_ran_off_end
  pure state

step_PROTO {n = S n'} bb state = do
  log "Opcode: PROTO"
  -- read a uint1
  (proto_ver, bb') <- bb_uncons bb
  putStr "Pickle protocol version: "
  printLn proto_ver
  step {n = n'} bb' state 

step_FRAME : {n : Nat} -> ByteBlock n -> VMState -> IO VMState
step_FRAME bb state = do
  log "Opcode: FRAME"
  case n of
    (S (S (S (S (S (S (S (S k)))))))) => do
      (frame_len, bb') <- read_uint8 bb
      putStr "Frame length is: "
      printLn frame_len
      putStr "Bytes remaining in buffer: "
      printLn (length bb')
      -- TODO: out of interest, validate FRAME against ByteBlock length.
      -- In pickle in general we can't do that because the input is a
      -- stream, not a fixed length block... but we know the length of
      -- ByteBlock because we're reading the whole thing before attempting
      -- a parse.
      
      step bb' state
    _ => ?error_not_enough_bytes_left_for_FRAME

step_MEMOIZE : {n: Nat} -> ByteBlock n -> VMState -> IO VMState
step_MEMOIZE bb (MkVMState (v::rest_stack) memo) = do
  log "Opcode: MEMOIZE"
  step bb (MkVMState (v::rest_stack) (memo ++ [v]))
step_MEMOIZE bb (MkVMState [] memo) = ?error_MEMOIZE_with_empty_stack


step_STOP : ByteBlock n -> VMState -> IO VMState
step_STOP bb state = do
  log "Opcode: STOP"
  pure state

-- The reasoning about lengths here is more complicated than PROTO or FRAME,
-- and maybe pushes more into runtime: the number of bytes we want is encoded
--  in the first remaining byte of the ByteBlock.
step_SHORT_BINUNICODE : {n: Nat} -> ByteBlock n -> VMState -> IO VMState
step_SHORT_BINUNICODE {n} bb (MkVMState stack memo) = do
  log "Opcode: SHORT_BINUNICODE"
  case n of
    (S k) => do
      (strlen, bb') <- bb_uncons bb
      putStr "UTF-8 byte sequence length: "
      printLn strlen

      -- the buffer contains strlen bytes of UTF-8 encoding, which we need to
      -- turn into an Idris2 String. could go via C, or could do it as a
      -- "read byte and append to idris2 string" entirely-in-idris impl?
      -- that latter is a bit more complicated - because a char and a byte are
      -- not the same thing. but we can "read a unicode codepoint" as
      -- a step (read bytes until its not a continuation byte?) and decode
      -- it into a char? that seems like a lot of faff compared to going via
      -- a C buffer...

      (str, bb'') <- str_from_bytes (cast strlen) bb'

      putStr "String is: "
      putStrLn str

      let new_state = MkVMState ((PickleUnicodeString str)::stack) memo

      step bb'' new_state

    Z => ?error_SHORT_BINUNICODE_no_length

step {n = Z} bb state = do
    log "ERROR: Pickle VM ran off end of pickle"
    ?error_pickle_ran_off_end

step {n = S m} bb state = do

    putStr "Stack pre-step: "
    printLn state.stack

    (opcode, bb') <- bb_uncons bb

    putStr "Opcode number: "
    printLn opcode

    case opcode of 
      46 => step_STOP bb' state
      128 => step_PROTO {n = m} bb' state
      140 => step_SHORT_BINUNICODE bb' state
      148 => step_MEMOIZE bb' state
      149 => step_FRAME {n = m} bb' state
      _ => ?error_unknown_opcode

||| Takes some representation of a byte sequence containing a pickle and
||| executes that pickle program, leaving something more like an AST of
||| the final object. That AST can be turned into an idris2 value by some
||| later function.
||| TODO: I'd like to be able to log pickle decoding without being embedded
||| in IO, so some effects system stuff? That maybe requires ByteBlock to be
||| immutable (which it isn't because ZMQ ops can edit it?) but maybe its
||| then interesting to flag ZMQ messages as frozen so they can't be
||| used with mutating ZMQ operations? Then are they safe to be used
||| functionally? maybe need some linear stuff on usage of msg object
||| to let that happen, session style?
export
unpickle : (n: Nat ** (ByteBlock n)) -> IO PickleAST
unpickle ((S n) ** bb) = do
  log "beginning unpickle"

  let init_vm_state = MkVMState [] []

  (MkVMState (stack_head::rest) end_memo) <- step bb init_vm_state
    | _ => ?error_stack_was_empty_at_end

  log "done with unpickle"
  pure stack_head

unpickle (Z ** bb) = do
  log "unpickle not defined on empty byte sequence"
  ?error_unpickle_Z


-- we can encode the length right in the type signature because
-- this is a simple opcode...
pickle_PROTO : Nat -> IO (ByteBlock 2)
pickle_PROTO v = do
  log "Pickling PROTO opcode"
  -- write PROTO byte
  -- write v as a byte

  -- can I do this symmetrically to bb_uncons, but appending
  -- at the end rather than taking off the front? i.e. treat
  -- ByteBlock as kinda looking like a stream with take off
  -- the front, append on the end... that probably fits the
  -- right sort of UI for a pickle buffer? (there's no need
  -- for the constructor to be a mirror of the deconstructor
  -- because this isn't an algebraic data type?)
  -- appending on the end probably maps ok-ish onto the
  -- API for realloc? except danger here! realloc can move
  -- the memory block, making any previous ByteBlocks that
  -- point into that space invalid... which is then maybe
  -- an interesting use for linear behaviour...
  -- my coding style so far has been quite linear with ByteBlocks
  -- so I wonder how easy/hard that will be to make into a
  -- linear style? also see that blog post on linear types with
  -- holes from tweag...
  -- so lets take the dangerous (without linear types) route
  -- and do reallocs...

  let bytes = emptyByteBlock

  bytes <- bb_append bytes 128   -- PROTO=128
  bytes <- bb_append bytes (cast v)

  pure bytes

||| Takes some PickleAST and turns it into a Pickle bytestream.
export
pickle : PickleAST -> IO (n: Nat ** (ByteBlock n))
pickle ast = do
  -- TODO: needs some kinds of ByteBlock access that we can write to in
  -- the ways that this code wants. I think that only means appending,
  -- because generally we will write out the args one by one (recursively)
  -- and then the relevant opcode and immediate arguments.

  -- 0-byte byte blocks seem a bit awkward in C land because malloc can return
  -- NULL if called with 0 size, so this would need some special casing?
  -- would be nice to have a monoidal structure (but in IO, I guess because
  -- of allocs? unless i start doing things with the ffi callback to free
  -- unused objects? which might fit here so that stuff is monoidal without
  -- any explicit effects (IO, linearity) ?) -- but I'd probably like to aim
  -- eventually for something linear to try to control resource usage rather
  -- than doing GC?

  -- PROTO 4
  proto_header_bytes <- pickle_PROTO 4

  --   cpython generated pickles then have a FRAME, but I think this isn't
  --   compulsory - although it might cause some performance degredation in
  --   streaming cases, I think in bytes-from-zmq cases like Parsl it won't
  --   matter as the block of bytes has already arrived and been placed into
  --   memory...

  -- then recursively run per-PickleAST-constructor code.

  ast_bytes <- ?encode_ast ast

  -- this could look like monoid expression syntax if this code was not tied
  -- into IO... appendBytes is <+> in IO (specifically, there are memory
  -- allocation semantics not just a pure value)
  bytes <- ?appendBytes proto_header_bytes ast_bytes

  pure bytes