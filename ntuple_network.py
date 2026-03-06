"""
N-Tuple Network for 2048 with Temporal Difference Learning.

Adapted from: https://github.com/moporgic/TDL2048-Demo

References:
  [1] M. Szubert and W. Jaśkowski, "Temporal difference learning of N-tuple
      networks for the game 2048," CIG 2014.
  [2] I-C. Wu et al., "Multi-stage temporal difference learning for 2048,"
      TAAI 2014.
"""
import sys
import math
import random
import struct
import array as _array
import abc


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

class board:
    """
    64-bit bitboard for 2048.

    Cell index layout (flat 0–15):
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15

    Each cell stores a 4-bit log₂ tile value:
        0 = empty, 1 = tile-2, 2 = tile-4, …, 15 = tile-32768
    """

    def __init__(self, raw: int = 0):
        self.raw = int(raw)

    def __int__(self) -> int:
        return self.raw

    def at(self, i: int) -> int:
        """Get the 4-bit log₂ value of cell i."""
        return (self.raw >> (i << 2)) & 0x0f

    def set(self, i: int, t: int) -> None:
        """Set the 4-bit log₂ value of cell i."""
        self.raw = (self.raw & ~(0x0f << (i << 2))) | ((t & 0x0f) << (i << 2))

    def fetch(self, i: int) -> int:
        """Get the 16-bit representation of row i."""
        return (self.raw >> (i << 4)) & 0xffff

    def place(self, i: int, r: int) -> None:
        """Set the 16-bit representation of row i."""
        self.raw = (self.raw & ~(0xffff << (i << 4))) | ((r & 0xffff) << (i << 4))

    def __getitem__(self, i: int) -> int:
        return self.at(i)

    def __setitem__(self, i: int, t: int) -> None:
        self.set(i, t)

    def __eq__(self, other) -> bool:
        return isinstance(other, board) and self.raw == other.raw

    def __ne__(self, other) -> bool:
        return not self == other

    def __lt__(self, other) -> bool:
        return isinstance(other, board) and self.raw < other.raw

    def __le__(self, other) -> bool:
        return isinstance(other, board) and not other < self

    def __gt__(self, other) -> bool:
        return isinstance(other, board) and other < self

    def __ge__(self, other) -> bool:
        return isinstance(other, board) and not self < other

    # ── Row-slide lookup table ────────────────────────────────────────────

    class lookup:
        find = [None] * 65536

        class entry:
            def __init__(self, row: int):
                V = [(row >> 0) & 0x0f, (row >> 4) & 0x0f,
                     (row >> 8) & 0x0f, (row >> 12) & 0x0f]
                L, score = board.lookup.entry._slide_left(V)
                Vr = V[::-1]
                R, _ = board.lookup.entry._slide_left(Vr)
                R = R[::-1]
                self.left  = L[0] | (L[1] << 4) | (L[2] << 8) | (L[3] << 12)
                self.right = R[0] | (R[1] << 4) | (R[2] << 8) | (R[3] << 12)
                self.score = score

            def move_left(self, raw: int, sc: int, i: int) -> tuple:
                return raw | (self.left << (i << 4)), sc + self.score

            def move_right(self, raw: int, sc: int, i: int) -> tuple:
                return raw | (self.right << (i << 4)), sc + self.score

            @staticmethod
            def _slide_left(row: list) -> tuple:
                buf = [t for t in row if t]
                res, score = [], 0
                while buf:
                    if len(buf) >= 2 and buf[0] == buf[1]:
                        buf = buf[1:]
                        buf[0] += 1
                        score += 1 << buf[0]
                        res.append(buf[0])
                    else:
                        res.append(buf[0])
                    buf = buf[1:]
                return res + [0] * (4 - len(res)), score

        @classmethod
        def init(cls) -> None:
            """Pre-compute slide results for all 65536 possible rows. Call once."""
            if cls.find[0] is None:
                cls.find = [cls.entry(row) for row in range(65536)]

    # ── Game actions ─────────────────────────────────────────────────────

    def init(self) -> None:
        """Reset to a new game: clear board and place two random tiles."""
        self.raw = 0
        self.popup()
        self.popup()

    def popup(self) -> None:
        """Place tile-2 (90%) or tile-4 (10%) in a random empty cell."""
        space = [i for i in range(16) if self.at(i) == 0]
        if space:
            self.set(random.choice(space), 1 if random.random() < 0.9 else 2)

    def move(self, opcode: int) -> int:
        """Apply action: 0=up, 1=right, 2=down, 3=left.
        Returns merge reward, or -1 if the action is illegal."""
        if opcode == 0:   return self.move_up()
        elif opcode == 1: return self.move_right()
        elif opcode == 2: return self.move_down()
        elif opcode == 3: return self.move_left()
        return -1

    def move_left(self) -> int:
        raw, score, prev = 0, 0, self.raw
        for i in range(4):
            raw, score = self.lookup.find[self.fetch(i)].move_left(raw, score, i)
        self.raw = raw
        return score if raw != prev else -1

    def move_right(self) -> int:
        raw, score, prev = 0, 0, self.raw
        for i in range(4):
            raw, score = self.lookup.find[self.fetch(i)].move_right(raw, score, i)
        self.raw = raw
        return score if raw != prev else -1

    def move_up(self) -> int:
        self.rotate_clockwise()
        score = self.move_right()
        self.rotate_counterclockwise()
        return score

    def move_down(self) -> int:
        self.rotate_clockwise()
        score = self.move_left()
        self.rotate_counterclockwise()
        return score

    # ── Board transformations (used for isomorphism generation) ──────────

    def transpose(self) -> None:
        r = self.raw
        r = ((r & 0xf0f00f0ff0f00f0f)
             | ((r & 0x0000f0f00000f0f0) << 12)
             | ((r & 0x0f0f00000f0f0000) >> 12))
        r = ((r & 0xff00ff0000ff00ff)
             | ((r & 0x00000000ff00ff00) << 24)
             | ((r & 0x00ff00ff00000000) >> 24))
        self.raw = r

    def mirror(self) -> None:
        r = self.raw
        self.raw = (((r & 0x000f000f000f000f) << 12)
                    | ((r & 0x00f000f000f000f0) << 4)
                    | ((r & 0x0f000f000f000f00) >> 4)
                    | ((r & 0xf000f000f000f000) >> 12))

    def flip(self) -> None:
        r = self.raw
        self.raw = (((r & 0x000000000000ffff) << 48)
                    | ((r & 0x00000000ffff0000) << 16)
                    | ((r & 0x0000ffff00000000) >> 16)
                    | ((r & 0xffff000000000000) >> 48))

    def rotate(self, r: int = 1) -> None:
        r = ((r % 4) + 4) % 4
        if r == 1:   self.rotate_clockwise()
        elif r == 2: self.reverse()
        elif r == 3: self.rotate_counterclockwise()

    def rotate_clockwise(self) -> None:
        self.transpose()
        self.mirror()

    def rotate_counterclockwise(self) -> None:
        self.transpose()
        self.flip()

    def reverse(self) -> None:
        self.mirror()
        self.flip()

    # ── Utilities ─────────────────────────────────────────────────────────

    def max_tile(self) -> int:
        """Return the log₂ value of the largest tile on the board."""
        return max(self.at(i) for i in range(16))

    def __str__(self) -> str:
        lines = ['+' + '-' * 24 + '+']
        for i in range(0, 16, 4):
            row = '|' + ''.join(
                f'{(1 << self.at(j)) & -2:6d}' for j in range(i, i + 4)
            ) + '|'
            lines.append(row)
        lines.append('+' + '-' * 24 + '+')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Feature (abstract base)
# ---------------------------------------------------------------------------

class feature(abc.ABC):
    """Abstract base for n-tuple feature weight tables."""

    @abc.abstractmethod
    def estimate(self, b: board) -> float:
        """Return the estimated value contribution for board b."""

    @abc.abstractmethod
    def update(self, b: board, u: float) -> float:
        """Update weights for board b by u and return the new estimate."""

    @abc.abstractmethod
    def name(self) -> str:
        """Return a unique name string (used as the binary file identifier)."""

    def size(self) -> int:
        return len(self.weight)  # works for both array.array and list

    def write(self, out) -> None:
        name_bytes = self.name().encode('utf-8')
        out.write(struct.pack('I', len(name_bytes)))
        out.write(name_bytes)
        out.write(struct.pack('Q', len(self.weight)))
        out.write(self.weight.tobytes())  # efficient C-level serialisation

    def read(self, inp) -> None:
        name_len = struct.unpack('I', inp.read(4))[0]
        stored_name = inp.read(name_len).decode('utf-8')
        if stored_name != self.name():
            print(f'feature mismatch: got {stored_name!r}, expected {self.name()!r}',
                  file=sys.stderr)
            sys.exit(1)
        size = struct.unpack('Q', inp.read(8))[0]
        if size != len(self.weight):
            print(f'weight size mismatch: {size} vs {len(self.weight)}', file=sys.stderr)
            sys.exit(1)
        self.weight = _array.array('f')
        self.weight.frombytes(inp.read(size * 4))


# ---------------------------------------------------------------------------
# Pattern (concrete n-tuple feature with isomorphic expansion)
# ---------------------------------------------------------------------------

class pattern(feature):
    """
    N-tuple pattern feature with isomorphic expansion.

    Each pattern is a list of cell indices (flat 0–15). All isomorphic
    variants (rotations + reflections) share a single weight table, which
    dramatically reduces the effective number of parameters while covering
    the full board symmetry group.

    Args:
        patt: Cell indices defining the base pattern, e.g. [0, 1, 2, 3, 4, 5].
        iso:  Isomorphism count — 8 (rotate+reflect, default), 4 (rotate only),
              or 1 (no symmetry sharing).

    Memory:
        16^len(patt) float32 entries, e.g. 64 MB for a 6-tuple.
    """

    def __init__(self, patt: list, iso: int = 8):
        n = 1 << (len(patt) * 4)
        # array.array('f') stores C float32 and returns Python floats on access —
        # ~5–10x faster scalar reads/writes than numpy for these hot-path lookups.
        self.weight = _array.array('f', bytes(n * 4))
        self._patt = patt
        # Precomputed bit-shift amounts: [0, 4, 8, 12, …] for 4-bit nibble packing.
        self._shifts = [4 * i for i in range(len(patt))]

        # Generate all isomorphic index-mapping lists by transforming a
        # reference board whose raw value encodes cell positions explicitly.
        self.isom: list[list[int]] = []
        for i in range(iso):
            idx = board(0xfedcba9876543210)
            if i >= 4:
                idx.mirror()
            idx.rotate(i)
            self.isom.append([idx.at(t) for t in patt])

    def estimate(self, b: board) -> float:
        # Extract all 16 tile values once (avoids N×8 repeated board.at() calls).
        raw    = b.raw
        tiles  = [(raw >> (i << 2)) & 0xf for i in range(16)]
        w      = self.weight
        shifts = self._shifts
        value  = 0.0
        for iso in self.isom:
            idx = 0
            for pos, sh in zip(iso, shifts):
                idx |= tiles[pos] << sh
            value += w[idx]
        return value

    def update(self, b: board, u: float) -> float:
        raw    = b.raw
        tiles  = [(raw >> (i << 2)) & 0xf for i in range(16)]
        adjust = u / len(self.isom)
        value  = 0.0
        w      = self.weight
        shifts = self._shifts
        for iso in self.isom:
            idx = 0
            for pos, sh in zip(iso, shifts):
                idx |= tiles[pos] << sh
            w[idx] += adjust
            value  += w[idx]
        return value

    def name(self) -> str:
        return (f"{len(self.isom[0])}-tuple pattern "
                f"{''.join(f'{p:x}' for p in self.isom[0])}")


# ---------------------------------------------------------------------------
# Move record
# ---------------------------------------------------------------------------

class move:
    """
    Record of a single game transition: state → action → afterstate.

    The afterstate is the board *after* sliding and merging, but *before*
    the random tile is placed. TD learning operates on afterstates so the
    value target is fully deterministic (no averaging over tile spawns).
    """

    def __init__(self, b: board = None, opcode: int = -1):
        self.before = None
        self.after  = None
        self.opcode = opcode
        self.score  = -1
        self.esti   = -math.inf
        if b is not None:
            self.assign(b)

    def afterstate(self) -> board:  return self.after
    def value(self)      -> float:  return self.esti
    def reward(self)     -> int:    return self.score
    def action(self)     -> int:    return self.opcode

    def set_value(self, v: float) -> None:
        self.esti = v

    def assign(self, b: board) -> bool:
        """Apply action to b, populate before/after/score/esti. Returns validity."""
        self.after  = board(b)
        self.before = board(b)
        self.score  = self.after.move(self.opcode)
        self.esti   = self.score if self.score != -1 else -math.inf
        return self.score != -1

    def is_valid(self) -> bool:
        if math.isnan(self.esti):
            sys.exit("N-Tuple: NaN detected — learning rate may be too high")
        return (self.after != self.before
                and self.opcode != -1
                and self.score  != -1)

    def __gt__(self, other) -> bool:
        return isinstance(other, move) and self.esti > other.esti


# ---------------------------------------------------------------------------
# Learning (the full network)
# ---------------------------------------------------------------------------

class learning:
    """
    N-tuple network: a collection of pattern features trained via TD(0)
    afterstate learning.

    Usage:
        net = learning()
        net.add_feature(pattern([0, 1, 2, 3, 4, 5]))
        # … add more features …
        net.load('ntuple.bin')        # no-op if file doesn't exist
        for game in ...:
            path = play_episode(net)
            net.learn_from_episode(path, alpha=0.1)
        net.save('ntuple.bin')
    """

    def __init__(self):
        self.feats: list[feature] = []

    def add_feature(self, feat: feature) -> None:
        self.feats.append(feat)
        mb = feat.size() * 4 / (1 << 20)
        print(f"  {feat.name()}, weights = {feat.size():,} ({mb:.0f} MB)")

    # ── Inference ─────────────────────────────────────────────────────────

    def estimate(self, b: board) -> float:
        """Sum of all feature estimates for board b."""
        return sum(f.estimate(b) for f in self.feats)

    def update(self, b: board, u: float) -> float:
        """Distribute update u across all features, return new estimate."""
        adjust = u / len(self.feats)
        return sum(f.update(b, adjust) for f in self.feats)

    def select_best_move(self, b: board) -> move:
        """
        Evaluate all four actions and return the move with the highest
        estimated value: reward(action) + V(afterstate).
        Returns an invalid sentinel move if no legal action exists.
        """
        best = move(b)  # sentinel: opcode=-1, esti=-inf
        for opcode in range(4):
            mv = move(b, opcode)
            if mv.is_valid():
                mv.set_value(mv.reward() + self.estimate(mv.afterstate()))
                if mv.value() > best.value():
                    best = mv
        return best

    # ── Learning ──────────────────────────────────────────────────────────

    def learn_from_episode(self, path: list, alpha: float = 0.1) -> None:
        """
        Backward TD(0) update through the episode.

        path: list of move records collected during play, including the
              terminal (invalid) move at the end.
        alpha: learning rate.

        Update rule (backward pass):
            target_T = 0
            for t = T-1 downto 0:
                error  = target_{t+1} - V(afterstate_t)
                target_t = reward_t + update(afterstate_t, alpha * error)
        """
        target = 0.0
        path.pop()  # discard terminal record (no valid afterstate)
        while path:
            mv = path.pop()
            error  = target - self.estimate(mv.afterstate())
            target = mv.reward() + self.update(mv.afterstate(), alpha * error)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save all feature weight tables to a binary file."""
        with open(path, 'wb') as f:
            f.write(struct.pack('Q', len(self.feats)))
            for feat in self.feats:
                feat.write(f)
        print(f"  weights saved → {path}")

    def load(self, path: str) -> bool:
        """Load feature weight tables from a binary file. Returns True on success."""
        try:
            with open(path, 'rb') as f:
                n = struct.unpack('Q', f.read(8))[0]
                if n != len(self.feats):
                    print(f'  feature count mismatch: file has {n}, '
                          f'expected {len(self.feats)}', file=sys.stderr)
                    return False
                for feat in self.feats:
                    feat.read(f)
            print(f"  weights loaded ← {path}")
            return True
        except FileNotFoundError:
            return False
