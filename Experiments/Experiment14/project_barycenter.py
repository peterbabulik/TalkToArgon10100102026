# requirements: numpy
# timeout: 300
"""
project_barycenter.py
=====================
Project BARYCENTER -- The Ontology of Zero
Multi-agent swarm execution: 4 specialist agents + Lead Architect orchestrator.

  AGENT 1  Low-Level Hardware & IEEE-754 Engineer   (signed zeros, 8-bit ALU, Zero Flag)
  AGENT 2  Hypercomplex Algebraist                  (Cayley-Dickson sedenion zero-divisors)
  AGENT 3  Quantum & Statistical Physicist          (Bloch-sphere barycenter, N = 1e6)
  AGENT 4  AI & Latent Space Architect              (Gram-Schmidt orthogonal blindspot)

Each experiment shows, in order:
  [INPUT STATE ENERGY]        the non-zero inputs exist in memory
  [EXECUTION / INTERFERENCE]  the operation that runs
  [OUTPUT STATE]              the zero that comes out
  [ONTOLOGICAL VERDICT]       why that zero is an event rather than a void

Only numpy plus the Python standard library (struct, ctypes, hashlib, math) is used.
The process exits with code 0 only if every experiment verifies.
"""

import sys
import os
import math
import time
import struct
import ctypes
import hashlib

import numpy as np

# ---------------------------------------------------------------------------
# Console setup
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Python 3.7+
except Exception:
    pass

if os.name == "nt":
    os.system("")  # enables ANSI escape processing on Windows 10+ terminals


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class C:
    _on = _color_enabled()
    RST = "\033[0m" if _on else ""
    BOLD = "\033[1m" if _on else ""
    DIM = "\033[2m" if _on else ""
    CYAN = "\033[96m" if _on else ""
    MAG = "\033[95m" if _on else ""
    GRN = "\033[92m" if _on else ""
    YEL = "\033[93m" if _on else ""
    RED = "\033[91m" if _on else ""
    BLU = "\033[94m" if _on else ""
    WHT = "\033[97m" if _on else ""


WIDTH = 92


def banner(title: str, subtitle: str = "") -> None:
    bar = "═" * WIDTH
    print(f"\n{C.MAG}{C.BOLD}╔{bar}╗{C.RST}")
    print(f"{C.MAG}{C.BOLD}║{C.RST} {C.CYAN}{C.BOLD}{title:<{WIDTH - 1}}{C.RST}{C.MAG}{C.BOLD}║{C.RST}")
    if subtitle:
        print(f"{C.MAG}{C.BOLD}║{C.RST} {C.DIM}{subtitle:<{WIDTH - 1}}{C.RST}{C.MAG}{C.BOLD}║{C.RST}")
    print(f"{C.MAG}{C.BOLD}╚{bar}╝{C.RST}")


def section(name: str, color: str = C.CYAN) -> None:
    pad = max(0, WIDTH - len(name) - 6)
    print(f"\n  {color}{C.BOLD}┌─[ {name} ]{'─' * pad}{C.RST}")


def kv(key: str, value, indent: int = 4) -> None:
    print(f"{' ' * indent}{C.DIM}»{C.RST} {C.WHT}{key:<40}{C.RST} {value}")


def txt(s: str, indent: int = 4, color: str = "") -> None:
    print(f"{' ' * indent}{color}{s}{C.RST if color else ''}")


def ok(flag: bool) -> str:
    return f"{C.GRN}{C.BOLD}[ VERIFIED ]{C.RST}" if flag else f"{C.RED}{C.BOLD}[ FAILED ]{C.RST}"


def verdict(lines) -> None:
    section("ONTOLOGICAL VERDICT", C.MAG)
    for ln in lines:
        txt(f"{C.MAG}▌{C.RST} {ln}")


# ===========================================================================
# AGENT 1 -- LOW-LEVEL HARDWARE & IEEE-754 ENGINEER
# ===========================================================================
class Agent1HardwareEngineer:
    """Signed zeros in IEEE-754 binary64 and an 8-bit ripple-carry ALU."""

    NAME = "AGENT 1 :: LOW-LEVEL HARDWARE & IEEE-754 ENGINEER"

    # ---------- IEEE-754 inspection -------------------------------------
    @staticmethod
    def bits_via_struct(x: float) -> int:
        return struct.unpack(">Q", struct.pack(">d", x))[0]

    @staticmethod
    def bits_via_ctypes(cell: ctypes.c_double) -> int:
        # Reinterpret the very same 8 bytes in RAM as an unsigned 64-bit integer.
        return ctypes.cast(ctypes.pointer(cell), ctypes.POINTER(ctypes.c_uint64)).contents.value

    @staticmethod
    def raw_ram_bytes(cell: ctypes.c_double) -> bytes:
        return ctypes.string_at(ctypes.addressof(cell), ctypes.sizeof(cell))

    @staticmethod
    def decompose(bits: int):
        sign = (bits >> 63) & 0x1
        exponent = (bits >> 52) & 0x7FF
        mantissa = bits & ((1 << 52) - 1)
        return sign, exponent, mantissa

    @staticmethod
    def fmt_bits64(bits: int) -> str:
        b = f"{bits:064b}"
        return f"{b[0]} {b[1:12]} {b[12:16]}…{b[-4:]}"

    # ---------- 8-bit ALU ------------------------------------------------
    @staticmethod
    def to_bits_lsb(v: int, width: int = 8):
        v &= (1 << width) - 1
        return [(v >> i) & 1 for i in range(width)]

    @staticmethod
    def bits_to_str_msb(bits_lsb) -> str:
        return "".join(str(b) for b in reversed(bits_lsb))

    @staticmethod
    def full_adder(a: int, b: int, cin: int):
        s = a ^ b ^ cin
        cout = (a & b) | (cin & (a ^ b))
        return s, cout

    def ripple_carry_add(self, x: int, y: int, width: int = 8):
        A = self.to_bits_lsb(x, width)
        B = self.to_bits_lsb(y, width)
        carry = 0
        S, trace = [], []
        carry_into_msb = 0
        for i in range(width):
            if i == width - 1:
                carry_into_msb = carry
            s, cout = self.full_adder(A[i], B[i], carry)
            trace.append((i, A[i], B[i], carry, s, cout, A[i] & B[i]))
            S.append(s)
            carry = cout
        cf = carry
        of = carry_into_msb ^ carry
        or_tree = 0
        for bit in S:
            or_tree |= bit
        zf = 1 - or_tree               # NOR across all register bits
        sf = S[-1]
        pf = 1 if (sum(S) % 2 == 0) else 0  # x86 PF: even parity of low byte
        return A, B, S, trace, {"CF": cf, "OF": of, "ZF": zf, "SF": sf, "PF": pf}

    # ---------- run -------------------------------------------------------
    def run(self) -> dict:
        banner(self.NAME, "Zero as a dual memory state and as an actively asserted CPU flag")

        pz_cell = ctypes.c_double(0.0)
        nz_cell = ctypes.c_double(-0.0)
        pz_struct, nz_struct = self.bits_via_struct(0.0), self.bits_via_struct(-0.0)
        pz_ct, nz_ct = self.bits_via_ctypes(pz_cell), self.bits_via_ctypes(nz_cell)

        # ---------------------------------------------------------------- INPUT
        section("INPUT STATE ENERGY")
        txt(f"{C.YEL}PART A -- IEEE-754 binary64 signed zeros resident in RAM{C.RST}")
        for label, cell, bs, bc in (("+0.0", pz_cell, pz_struct, pz_ct),
                                    ("-0.0", nz_cell, nz_struct, nz_ct)):
            s, e, m = self.decompose(bc)
            kv(f"{label} @ RAM address", f"0x{ctypes.addressof(cell):016X}")
            kv(f"{label} raw bytes (native order)", self.raw_ram_bytes(cell).hex(" ").upper())
            kv(f"{label} struct  >Q  bit pattern", f"0x{bs:016X}")
            kv(f"{label} ctypes uint64 reinterpret", f"0x{bc:016X}")
            kv(f"{label} [sign | exponent | mantissa]", f"[{s} | {e:011b} | {m:052b}]"[:62] + "…]")
            kv(f"{label} bit field  S EEEEEEEEEEE MMMM…", self.fmt_bits64(bc))

        txt("")
        txt(f"{C.YEL}PART B -- 8-bit operands loaded into ALU registers{C.RST}")
        a_val, b_val = 4, -4
        b_twos = ((~4) + 1) & 0xFF
        kv("Operand A = +4", f"{a_val & 0xFF:08b}  (0x{a_val & 0xFF:02X})")
        kv("Operand B = -4  (two's complement)", f"~00000100 + 1 = {(~4) & 0xFF:08b} + 1 = {b_twos:08b}  (0x{b_twos:02X})")
        kv("Set bits in A / B (non-zero energy)", f"{bin(a_val & 0xFF).count('1')} / {bin(b_val & 0xFF).count('1')} high bits")

        # ------------------------------------------------------------ EXECUTION
        section("EXECUTION / INTERFERENCE")
        eq = (0.0 == -0.0)
        kv("Python:  0.0 == -0.0", f"{C.GRN}{eq}{C.RST}")
        kv("Bit patterns identical?", f"{C.RED}{pz_ct == nz_ct}{C.RST}  (XOR = 0x{pz_ct ^ nz_ct:016X})")
        kv("math.copysign(1.0, +0.0 / -0.0)", f"{math.copysign(1.0, 0.0):+.1f} / {math.copysign(1.0, -0.0):+.1f}")
        with np.errstate(divide="ignore"):
            inv_p = np.float64(1.0) / np.float64(0.0)
            inv_n = np.float64(1.0) / np.float64(-0.0)
        kv("1 / +0.0  vs  1 / -0.0 (numpy)", f"{inv_p}  vs  {inv_n}")
        kv("atan2(+0.0, -1)  vs  atan2(-0.0, -1)", f"{math.atan2(0.0, -1.0):+.6f}  vs  {math.atan2(-0.0, -1.0):+.6f}")

        txt("")
        txt(f"{C.YEL}Ripple-carry full-adder chain  (S = A⊕B⊕Cin, Cout = AB + Cin(A⊕B)){C.RST}")
        A, B, S, trace, flags = self.ripple_carry_add(a_val, b_val)
        hdr = f"{'bit':>5} │ {'A':>2} {'B':>2} │ {'Cin':>4} │ {'Sum':>4} │ {'Cout':>5} │ event"
        txt(f"{C.DIM}{hdr}{C.RST}", indent=6)
        txt(f"{C.DIM}{'─' * len(hdr)}{C.RST}", indent=6)
        for i, a, b, cin, s, cout, collide in trace:
            if collide:
                ev = f"{C.RED}COLLISION 1+1 → 0, carry 1{C.RST}"
            elif cin and (a ^ b):
                ev = f"{C.YEL}carry absorbed: 1+carry → 0, carry 1{C.RST}"
            else:
                ev = f"{C.DIM}quiet{C.RST}"
            if i == 7 and cout:
                ev += f"  {C.RED}{C.BOLD}→ EJECTED FROM REGISTER{C.RST}"
            txt(f"{i:>5} │ {a:>2} {b:>2} │ {cin:>4} │ {s:>4} │ {cout:>5} │ {ev}", indent=6)

        # --------------------------------------------------------------- OUTPUT
        section("OUTPUT STATE")
        reg = self.bits_to_str_msb(S)
        kv("    A", self.bits_to_str_msb(A))
        kv("  + B", self.bits_to_str_msb(B))
        kv("  = 9-bit physical result (CF:REG)", f"{flags['CF']}:{reg}")
        kv("8-bit register contents", f"{C.CYAN}{C.BOLD}{reg}{C.RST}  (value {int(reg, 2)})")
        kv("Carry Flag  CF (bit ejected)", f"{C.RED}{C.BOLD}{flags['CF']}{C.RST}")
        kv("Overflow Flag OF (Cin7 ⊕ Cout7)", f"{flags['OF']}  (signed result is valid)")
        kv("Zero Flag ZF = NOR(r7..r0)", f"{C.GRN}{C.BOLD}{flags['ZF']}{C.RST}  ← a physical HIGH latch")
        kv("Sign Flag SF / Parity Flag PF", f"{flags['SF']} / {flags['PF']}")
        eflags = (flags["CF"] << 0) | (1 << 1) | (flags["PF"] << 2) | (flags["ZF"] << 6) | \
                 (flags["SF"] << 7) | (flags["OF"] << 11)
        kv("x86-style FLAGS word", f"0x{eflags:04X} = {eflags:016b}b  (bit1 reserved=1)")

        passed = (eq and pz_struct == 0x0000000000000000 and nz_struct == 0x8000000000000000
                  and pz_ct == pz_struct and nz_ct == nz_struct
                  and reg == "00000000" and flags["CF"] == 1 and flags["ZF"] == 1 and flags["OF"] == 0)
        kv("Experiment status", ok(passed))

        verdict([
            "IEEE-754 stores two distinct zeros: 0x0000000000000000 (+0) and 0x8000000000000000 (-0).",
            "Comparison calls them equal, but division and atan2 tell them apart by their direction.",
            "+4 and -4 collide bit by bit. The carry ripples through the chain, the overflow bit is",
            "ejected (CF=1), and the 8-bit register is left at 00000000.",
            "The CPU records this result by driving a NOR gate HIGH and latching ZF = 1.",
            "So at the hardware level, zero is an asserted state: the residue of an interference event.",
        ])
        return {"agent": "A1 Hardware/IEEE-754", "zero": f"REG={reg}, ZF={flags['ZF']}", "passed": passed}


# ===========================================================================
# AGENT 2 -- HYPERCOMPLEX ALGEBRAIST (SEDENION ZERO-DIVISOR)
# ===========================================================================
class Agent2HypercomplexAlgebraist:
    """Recursive Cayley-Dickson algebra R → C → H → O → S (16D)."""

    NAME = "AGENT 2 :: HYPERCOMPLEX ALGEBRAIST (SEDENION ZERO-DIVISOR)"
    DIM = 16
    CONVENTION = "(a,b)·(c,d) = (a·c − d*·b ,  d·a + b·c*),   (a,b)* = (a*, −b)"

    # ---------- Cayley-Dickson core -------------------------------------
    @staticmethod
    def cd_conj(x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 1:
            return x.copy()
        h = n // 2
        return np.concatenate((Agent2HypercomplexAlgebraist.cd_conj(x[:h]), -x[h:]))

    @staticmethod
    def cd_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n != y.shape[0] or (n & (n - 1)) != 0:
            raise ValueError("Cayley-Dickson operands must share a power-of-two dimension")
        if n == 1:
            return x * y
        h = n // 2
        a, b = x[:h], x[h:]
        c, d = y[:h], y[h:]
        conj = Agent2HypercomplexAlgebraist.cd_conj
        mul = Agent2HypercomplexAlgebraist.cd_mul
        left = mul(a, c) - mul(conj(d), b)
        right = mul(d, a) + mul(b, conj(c))
        return np.concatenate((left, right))

    @classmethod
    def basis(cls, i: int, dim: int = None) -> np.ndarray:
        dim = dim or cls.DIM
        v = np.zeros(dim, dtype=np.float64)
        v[i] = 1.0
        return v

    @classmethod
    def sparse(cls, terms) -> np.ndarray:
        v = np.zeros(cls.DIM, dtype=np.float64)
        for idx, coef in terms:
            v[idx] += coef
        return v

    @staticmethod
    def fmt_terms(terms) -> str:
        out = []
        for k, (idx, coef) in enumerate(terms):
            sign = "+" if coef > 0 else "−"
            lead = ("" if sign == "+" else "−") if k == 0 else f" {sign} "
            out.append(f"{lead}e{idx}")
        return "".join(out)

    @staticmethod
    def fmt_vec16(v: np.ndarray):
        cells = [f"{x:+.1f}" for x in v]
        return ("[" + ", ".join(cells[:8]) + ",", " " + ", ".join(cells[8:]) + "]")

    def structure_tensor(self) -> np.ndarray:
        M = np.zeros((self.DIM, self.DIM, self.DIM), dtype=np.float64)
        for i in range(self.DIM):
            ei = self.basis(i)
            for j in range(self.DIM):
                M[i, j, :] = self.cd_mul(ei, self.basis(j))
        return M

    # ---------- algebra self-tests ---------------------------------------
    def self_tests(self, rng: np.random.Generator) -> dict:
        res = {}
        res["imag_units_square_to_-1"] = all(
            np.array_equal(self.cd_mul(self.basis(i), self.basis(i)), -self.basis(0)) for i in range(1, 16))
        res["e0_is_identity"] = all(
            np.array_equal(self.cd_mul(self.basis(0), self.basis(i)), self.basis(i)) for i in range(16))
        errs = []
        for _ in range(200):
            x, y = rng.normal(size=8), rng.normal(size=8)
            errs.append(abs(np.linalg.norm(self.cd_mul(x, y)) - np.linalg.norm(x) * np.linalg.norm(y)))
        res["octonion_norm_multiplicative_maxerr"] = max(errs)
        res["octonion_composition_ok"] = max(errs) < 1e-10
        return res

    # ---------- zero-divisor discovery -----------------------------------
    def find_zero_divisor(self):
        tried = []
        candidates = [
            ([(3, 1.0), (10, 1.0)], [(6, 1.0), (15, -1.0)]),   # requested canonical form
            ([(3, 1.0), (10, 1.0)], [(6, 1.0), (15, 1.0)]),
            ([(3, 1.0), (10, -1.0)], [(6, 1.0), (15, -1.0)]),
            ([(3, 1.0), (10, -1.0)], [(6, 1.0), (15, 1.0)]),
        ]
        for ta, tb in candidates:
            A, B = self.sparse(ta), self.sparse(tb)
            n = float(np.linalg.norm(self.cd_mul(A, B)))
            tried.append((self.fmt_terms(ta), self.fmt_terms(tb), n))
            if n == 0.0:
                return ta, tb, tried, "requested family (e3 ± e10)(e6 ± e15)"

        # Exhaustive fallback over all (e_a ± e_b), 1 ≤ a < b ≤ 15, vectorised.
        M = self.structure_tensor()
        terms_list, rows = [], []
        for a in range(1, 16):
            for b in range(a + 1, 16):
                for s in (1.0, -1.0):
                    t = [(a, 1.0), (b, s)]
                    terms_list.append(t)
                    rows.append(self.sparse(t))
        V = np.array(rows)
        T = np.einsum("pi,ijk->pjk", V, M)
        R = np.einsum("pjk,qj->pqk", T, V)
        norms = np.linalg.norm(R, axis=2)
        hits = np.argwhere(norms == 0.0)
        if hits.size == 0:
            raise RuntimeError("No two-term sedenion zero-divisor found (should be impossible).")
        p, q = hits[0]
        return terms_list[p], terms_list[q], tried, "exhaustive two-term search"

    # ---------- run -------------------------------------------------------
    def run(self, rng: np.random.Generator) -> dict:
        banner(self.NAME, "Beyond Hurwitz: two non-zero 16D objects multiply to exactly zero")

        tests = self.self_tests(rng)
        ta, tb, tried, source = self.find_zero_divisor()
        A, B = self.sparse(ta), self.sparse(tb)
        nA, nB = float(np.linalg.norm(A)), float(np.linalg.norm(B))

        # A is individually invertible: A · (A*/|A|²) = 1
        A_inv = self.cd_conj(A) / (nA ** 2)
        B_inv = self.cd_conj(B) / (nB ** 2)
        AAinv = self.cd_mul(A, A_inv)
        BBinv = self.cd_mul(B, B_inv)

        # ---------------------------------------------------------------- INPUT
        section("INPUT STATE ENERGY")
        kv("Construction", "recursive Cayley-Dickson  R(1)→C(2)→H(4)→O(8)→S(16)")
        kv("Doubling convention", self.CONVENTION)
        kv("e_i² = −1 for i = 1..15", ok(tests["imag_units_square_to_-1"]))
        kv("e0 acts as identity", ok(tests["e0_is_identity"]))
        kv("Octonion |xy| = |x||y| (200 trials)", f"max err {tests['octonion_norm_multiplicative_maxerr']:.2e}  "
                                                  f"{ok(tests['octonion_composition_ok'])}")
        kv("Zero-divisor source", source)
        for fa, fb, n in tried:
            kv(f"  probe ({fa})·({fb})", f"||·|| = {n:.6f}")
        txt("")
        kv("A", f"{C.CYAN}{self.fmt_terms(ta)}{C.RST}")
        r1, r2 = self.fmt_vec16(A)
        kv("A components", r1); kv("", r2)
        kv("||A||", f"{nA:.15f}  (√2 = {math.sqrt(2):.15f})")
        kv("B", f"{C.CYAN}{self.fmt_terms(tb)}{C.RST}")
        r1, r2 = self.fmt_vec16(B)
        kv("B components", r1); kv("", r2)
        kv("||B||", f"{nB:.15f}")
        kv("Energy in  ||A||² + ||B||²", f"{nA ** 2 + nB ** 2:.6f}")
        kv("A·A⁻¹ = e0 ?  (A is invertible alone)", ok(np.allclose(AAinv, self.basis(0), atol=1e-15)))
        kv("B·B⁻¹ = e0 ?  (B is invertible alone)", ok(np.allclose(BBinv, self.basis(0), atol=1e-15)))

        # ------------------------------------------------------------ EXECUTION
        section("EXECUTION / INTERFERENCE")
        txt(f"C = A × B = ({self.fmt_terms(ta)}) × ({self.fmt_terms(tb)})", color=C.YEL)
        txt("Expanded into the four basis products (structure constants of S):", color=C.DIM)
        partials = []
        for ia, ca in ta:
            for ib, cb in tb:
                prod = ca * cb * self.cd_mul(self.basis(ia), self.basis(ib))
                k = int(np.argmax(np.abs(prod)))
                sgn = prod[k]
                partials.append((ia, ca, ib, cb, k, sgn))
                kv(f"  ({ca:+.0f}e{ia})·({cb:+.0f}e{ib})", f"{sgn:+.0f}·e{k}")
        cancel = {}
        for *_, k, s in partials:
            cancel[k] = cancel.get(k, 0.0) + s
        kv("Per-axis interference sums", ", ".join(f"e{k}: {v:+.0f}" for k, v in sorted(cancel.items())))

        Cv = self.cd_mul(A, B)
        nC = float(np.linalg.norm(Cv))

        # --------------------------------------------------------------- OUTPUT
        section("OUTPUT STATE")
        r1, r2 = self.fmt_vec16(Cv)
        kv("C = A × B  (16D)", f"{C.CYAN}{r1}{C.RST}")
        kv("", f"{C.CYAN}{r2}{C.RST}")
        kv("||A × B||", f"{C.GRN}{C.BOLD}{nC!r}{C.RST}  (exact == 0.0: {nC == 0.0})")
        kv("||A||·||B||  (composition law prediction)", f"{nA * nB:.6f}  → composition law broken")

        # Information loss / entropy spike
        L_A = np.column_stack([self.cd_mul(A, self.basis(j)) for j in range(16)])
        rank = int(np.linalg.matrix_rank(L_A, tol=1e-12))
        ker = 16 - rank
        pA = (A ** 2) / np.sum(A ** 2)
        H_A = float(-np.sum(pA[pA > 0] * np.log2(pA[pA > 0])))
        txt("")
        txt(f"{C.RED}{C.BOLD}⚠ INFORMATION LOSS / ENTROPY SPIKE LOG{C.RST}")
        kv("Energy in  → energy out", f"{nA ** 2 + nB ** 2:.4f} → {nC ** 2:.4f}   (100.000% annihilated)")
        kv("Left-multiplication operator L_A rank", f"{rank} / 16")
        kv("ker(L_A) dimension", f"{C.RED}{ker}{C.RST} real degrees of freedom of B are erased")
        kv("B ∈ ker(L_A) ?", ok(bool(np.allclose(L_A @ B, 0.0))))
        kv("Shannon entropy of A's energy spread", f"{H_A:.3f} bit  → output: undefined (no energy)")
        kv("Reversibility", "A×B = 0 cannot be inverted to recover B (singular collapse)")

        passed = (nC == 0.0 and abs(nA - math.sqrt(2)) < 1e-15 and abs(nB - math.sqrt(2)) < 1e-15
                  and tests["imag_units_square_to_-1"] and tests["octonion_composition_ok"] and ker > 0)
        kv("Experiment status", ok(passed))

        verdict([
            "A and B each have norm √2 and each has its own inverse, so both fully exist.",
            "Hurwitz: composition algebras stop at the octonions (8D), and the composition law",
            "|xy| = |x||y| above was verified for 8D. In 16D sedenions that law breaks.",
            "Their basis products fall pairwise onto the same axes with opposite signs and cancel.",
            f"Product = exact zero. {ker} dimensions of information about B collapse into a singularity.",
            "This zero is constructive in reverse: destructive interference of two non-zero beings.",
        ])
        return {"agent": "A2 Sedenion Zero-Divisor",
                "zero": f"||({self.fmt_terms(ta)})({self.fmt_terms(tb)})|| = {nC}", "passed": passed}


# ===========================================================================
# AGENT 3 -- QUANTUM & STATISTICAL PHYSICIST (BLOCH SPHERE BARYCENTER)
# ===========================================================================
class Agent3StatisticalPhysicist:
    """Macroscopic zero hiding microscopic high-energy chaos."""

    NAME = "AGENT 3 :: QUANTUM & STATISTICAL PHYSICIST (BLOCH BARYCENTER)"
    N = 1_000_000
    AMP = 4.0
    SIGMA_T = 0.5   # thermal noise std

    def run(self, rng: np.random.Generator) -> dict:
        banner(self.NAME, "The centre of the Bloch sphere: a zero made of a million ±4 oscillators")
        N, AMP, SIG = self.N, self.AMP, self.SIGMA_T
        t0 = time.perf_counter()

        # Ensemble I: independent random spins ±4 with thermal noise
        signs = rng.choice(np.array([-1.0, 1.0]), size=N)
        thermal = rng.normal(0.0, SIG, size=N)
        psi = AMP * signs + thermal

        # Ensemble II: antithetic (mirrored) ensemble -- every oscillator has a partner
        half = N // 2
        h = AMP * rng.choice(np.array([-1.0, 1.0]), size=half) + rng.normal(0.0, SIG, size=half)
        psi_pair = np.empty(N, dtype=np.float64)
        psi_pair[0::2] = h
        psi_pair[1::2] = -h

        # Ensemble III: 1e6 pure-state Bloch vectors, isotropically distributed
        r = rng.normal(size=(N, 3))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        gen_ms = (time.perf_counter() - t0) * 1e3

        # ---------------------------------------------------------------- INPUT
        section("INPUT STATE ENERGY")
        msa = float(np.mean(psi ** 2))
        l2 = float(np.linalg.norm(psi))
        total_E = float(np.sum(psi ** 2))
        kv("Oscillators N", f"{N:,}")
        kv("Micro-state model", f"ψ_i = ±{AMP:.0f} + η_i,   η ~ N(0, {SIG}²)")
        kv("Tensor memory footprint", f"{psi.nbytes / 2 ** 20:.2f} MiB (float64)  + Bloch {r.nbytes / 2 ** 20:.2f} MiB")
        kv("Generation time", f"{gen_ms:.1f} ms")
        kv("Non-zero oscillators", f"{int(np.count_nonzero(psi)):,} / {N:,}")
        kv("Mean |ψ_i|", f"{float(np.mean(np.abs(psi))):.6f}")
        kv("System energy  <ψ²>  (mean sq. amp.)", f"{C.RED}{C.BOLD}{msa:.6f}{C.RST}  (theory {AMP ** 2 + SIG ** 2:.4f})")
        kv("Total energy   Σψ²", f"{total_E:,.2f}")
        kv("L2 norm  ||ψ||₂", f"{l2:,.4f}")
        kv("RMS amplitude", f"{math.sqrt(msa):.6f}")

        # ------------------------------------------------------------ EXECUTION
        section("EXECUTION / INTERFERENCE")
        txt("Macroscopic observation: expectation value <ψ> = (1/n) Σ ψ_i   (law of large numbers)",
            color=C.YEL)
        hdr = f"{'n':>10} │ {'<ψ>_n':>14} │ {'|<ψ>|·√n':>10} │ {'std.err σ/√n':>13}"
        txt(f"{C.DIM}{hdr}{C.RST}", indent=6)
        txt(f"{C.DIM}{'─' * len(hdr)}{C.RST}", indent=6)
        csum = np.cumsum(psi)
        rms = math.sqrt(msa)
        for e in range(1, 7):
            n = 10 ** e
            m = csum[n - 1] / n
            txt(f"{n:>10,} │ {m:>+14.8f} │ {abs(m) * math.sqrt(n):>10.4f} │ {rms / math.sqrt(n):>13.8f}", indent=6)
        txt("|<ψ>|·√n stays O(σ) → <ψ> decays as 1/√n → 0 as N → ∞", color=C.DIM, indent=6)

        mean_rand = float(np.mean(psi))
        mean_pair_exact = math.fsum(psi_pair) / N
        rbar = r.mean(axis=0)
        rbar_norm = float(np.linalg.norm(rbar))
        I2 = np.eye(2, dtype=complex)
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        rho = 0.5 * (I2 + rbar[0] * sx + rbar[1] * sy + rbar[2] * sz)
        purity = float(np.real(np.trace(rho @ rho)))
        kv("Bloch ensemble: every |r_i| = 1 ?", ok(bool(np.allclose(np.linalg.norm(r[:1000], axis=1), 1.0))))
        kv("Ensemble density matrix ρ = ½(I + r̄·σ)", "")
        txt(f"[[{rho[0, 0].real:+.6f}{rho[0, 0].imag:+.6f}j, {rho[0, 1].real:+.6f}{rho[0, 1].imag:+.6f}j],", indent=10)
        txt(f" [{rho[1, 0].real:+.6f}{rho[1, 0].imag:+.6f}j, {rho[1, 1].real:+.6f}{rho[1, 1].imag:+.6f}j]]", indent=10)

        # --------------------------------------------------------------- OUTPUT
        section("OUTPUT STATE")
        kv("Ensemble I   <ψ>  (random spins)", f"{C.GRN}{mean_rand:+.10f}{C.RST}   ≈ 0  (|.| < 5σ/√N: "
                                                 f"{abs(mean_rand) < 5 * rms / math.sqrt(N)})")
        kv("Ensemble II  <ψ>  (mirrored pairs, fsum)", f"{C.GRN}{C.BOLD}{mean_pair_exact!r}{C.RST}  (exact zero)")
        kv("Ensemble II  <ψ²>", f"{float(np.mean(psi_pair ** 2)):.6f}  (same energy, zero signal)")
        kv("Bloch barycenter |r̄|", f"{C.GRN}{rbar_norm:.3e}{C.RST}  → centre of sphere")
        kv("Purity Tr(ρ²)", f"{purity:.8f}  (maximally mixed = 0.5)")
        ratio_E = msa / abs(mean_rand)
        ratio_rms = rms / abs(mean_rand)
        txt("")
        kv("RATIO  <ψ²> / |<ψ>|   (Ensemble I)", f"{C.RED}{C.BOLD}{ratio_E:,.1f} : 1{C.RST}")
        kv("RATIO  RMS / |<ψ>|    (Ensemble I)", f"{ratio_rms:,.1f} : 1")
        kv("RATIO  <ψ²> / |<ψ>|   (Ensemble II)",
           f"{C.RED}{C.BOLD}∞{C.RST}  (finite energy {float(np.mean(psi_pair ** 2)):.3f} / observed 0.0)")

        passed = (abs(mean_rand) < 5 * rms / math.sqrt(N) and mean_pair_exact == 0.0
                  and msa > 15.0 and rbar_norm < 5.0 / math.sqrt(N) and abs(purity - 0.5) < 1e-4)
        kv("Experiment status", ok(passed))

        verdict([
            f"Every one of the {N:,} oscillators carries energy ≈ {msa:.2f}; none of them is zero.",
            "The macroscopic observable <ψ> still converges to 0. It is a cancellation, not an absence.",
            "One million pure Bloch states (each |r| = 1) average to the centre of the sphere.",
            "That centre is ρ = I/2, the maximally mixed state with purity 0.5.",
            f"Energy outweighs the observed signal by {ratio_E:,.0f}:1, and by ∞:1 for mirrored pairs.",
            "Zero is the barycenter of a storm: a statistical mask over maximal microscopic activity.",
        ])
        return {"agent": "A3 Bloch Barycenter",
                "zero": f"<ψ>={mean_rand:+.2e} | paired={mean_pair_exact} | <ψ²>={msa:.2f}", "passed": passed}


# ===========================================================================
# AGENT 4 -- AI & LATENT SPACE ARCHITECT (ORTHOGONAL BLINDSPOT)
# ===========================================================================
class Agent4LatentSpaceArchitect:
    """Zero activation as 90° geometry between two fully-present embeddings."""

    NAME = "AGENT 4 :: AI & LATENT SPACE ARCHITECT (ORTHOGONAL BLINDSPOT)"
    D = 512

    @staticmethod
    def sha(v: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()

    @staticmethod
    def gram_schmidt(u: np.ndarray, v: np.ndarray, passes: int = 2):
        """Returns (e1, e2): e1 = u/|u|, e2 = component of v ⟂ e1, normalised.
        Classical Gram-Schmidt with re-orthogonalisation ("twice is enough")."""
        e1 = u / np.linalg.norm(u)
        w = v.astype(np.float64).copy()
        for _ in range(passes):
            w = w - np.dot(w, e1) * e1
        e2 = w / np.linalg.norm(w)
        return e1, e2

    def run(self, rng: np.random.Generator) -> dict:
        banner(self.NAME, "A neuron that outputs 0 is not looking at nothing; it is looking at 90°")
        D = self.D
        raw_in = rng.normal(size=D)
        raw_w = rng.normal(size=D)
        raw_cos = float(np.dot(raw_in, raw_w) / (np.linalg.norm(raw_in) * np.linalg.norm(raw_w)))

        Input_Vector, Weight_Vector = self.gram_schmidt(raw_in, raw_w)
        Input_Vector.setflags(write=False)
        Weight_Vector.setflags(write=False)

        sha_in_before, sha_w_before = self.sha(Input_Vector), self.sha(Weight_Vector)
        ptr_in = Input_Vector.__array_interface__["data"][0]
        ptr_w = Weight_Vector.__array_interface__["data"][0]

        # ---------------------------------------------------------------- INPUT
        section("INPUT STATE ENERGY")
        kv("Embedding dimension", f"{D}-D float64")
        kv("Input_Vector  @ RAM", f"0x{ptr_in:016X}   ({Input_Vector.nbytes} bytes)")
        kv("Weight_Vector @ RAM", f"0x{ptr_w:016X}   ({Weight_Vector.nbytes} bytes)")
        kv("||Input_Vector||₂", f"{C.CYAN}{np.linalg.norm(Input_Vector):.15f}{C.RST}")
        kv("||Weight_Vector||₂", f"{C.CYAN}{np.linalg.norm(Weight_Vector):.15f}{C.RST}")
        kv("Non-zero components (in / w)", f"{np.count_nonzero(Input_Vector)} / {np.count_nonzero(Weight_Vector)}  of {D}")
        kv("max|x_i|  /  max|w_i|", f"{np.max(np.abs(Input_Vector)):.6f}  /  {np.max(np.abs(Weight_Vector)):.6f}")
        kv("Input_Vector[:4]", np.array2string(Input_Vector[:4], precision=6, separator=", "))
        kv("Weight_Vector[:4]", np.array2string(Weight_Vector[:4], precision=6, separator=", "))
        kv("SHA-256(Input_Vector)", sha_in_before[:48] + "…")
        kv("SHA-256(Weight_Vector)", sha_w_before[:48] + "…")

        # ------------------------------------------------------------ EXECUTION
        section("EXECUTION / INTERFERENCE")
        kv("Pre-orthogonalisation cosine", f"{raw_cos:+.6f}  (random 512-D vectors ≈ ±1/√512)")
        kv("Gram-Schmidt", "w ← w − (w·x̂)x̂   ×2 passes, then w ← w/||w||")
        W = Weight_Vector.reshape(1, D)          # Linear layer: 1 neuron, 512 inputs
        bias = np.zeros(1)
        z = W @ Input_Vector + bias              # pre-activation
        activation = float(z[0])
        relu = max(0.0, activation)
        exact_sum = math.fsum(float(a) * float(b) for a, b in zip(Weight_Vector, Input_Vector))
        cos_theta = activation / (np.linalg.norm(Weight_Vector) * np.linalg.norm(Input_Vector))
        theta = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
        kv("Linear layer  z = W·x + b", f"W∈R^(1×{D}), b = 0")
        kv("Element-wise products Σ|w_i x_i|", f"{float(np.sum(np.abs(Weight_Vector * Input_Vector))):.6f}  (signal flowing)")
        kv("Positive / negative contributions",
           f"{float(np.sum(np.clip(Weight_Vector * Input_Vector, 0, None))):+.6f} / "
           f"{float(np.sum(np.clip(Weight_Vector * Input_Vector, None, 0))):+.6f}")

        # Contrast: a non-orthogonal filter
        contrast_w = rng.normal(size=D)
        contrast_w /= np.linalg.norm(contrast_w)
        contrast_act = float(np.dot(contrast_w, Input_Vector))

        # --------------------------------------------------------------- OUTPUT
        section("OUTPUT STATE")
        kv("Activation z = W·x (numpy)", f"{C.GRN}{C.BOLD}{activation:+.3e}{C.RST}")
        kv("Activation (fsum, compensated)", f"{exact_sum:+.3e}")
        kv("Machine epsilon (float64)", f"{np.finfo(np.float64).eps:.3e}")
        kv("ReLU(z)", f"{relu:.3e}")
        kv("Angle θ between x and w", f"{C.CYAN}{theta:.12f}°{C.RST}")
        kv("Contrast: random unit filter · x", f"{contrast_act:+.6f}  (non-zero: not orthogonal)")

        sha_in_after, sha_w_after = self.sha(Input_Vector), self.sha(Weight_Vector)
        intact = (sha_in_after == sha_in_before and sha_w_after == sha_w_before
                  and abs(np.linalg.norm(Input_Vector) - 1.0) < 1e-12
                  and abs(np.linalg.norm(Weight_Vector) - 1.0) < 1e-12)
        kv("Post-op SHA-256 match (in / w)", f"{sha_in_after == sha_in_before} / {sha_w_after == sha_w_before}")
        kv("Both actors intact in RAM", ok(intact))

        passed = intact and abs(activation) < 1e-12 and abs(theta - 90.0) < 1e-9
        kv("Experiment status", ok(passed))

        verdict([
            "Both embeddings have unit norm and 512/512 non-zero components.",
            "Their SHA-256 hashes are unchanged after the forward pass.",
            "Signal does flow: hundreds of products w_i·x_i fire, and their + and − halves cancel.",
            f"The activation {activation:+.1e} is machine-epsilon zero because the vectors meet at exactly 90°.",
            "In latent space, 0 means 'no correlation' — a geometric relation between two full",
            "objects. The neuron is blind to the data, not empty of it: an orthogonal blindspot.",
        ])
        return {"agent": "A4 Orthogonal Blindspot",
                "zero": f"W·x = {activation:+.2e}, θ = {theta:.9f}°", "passed": passed}


# ===========================================================================
# LEAD ARCHITECT -- ORCHESTRATOR
# ===========================================================================
class LeadArchitect:
    SEED = 0xBA2C3E7  # deterministic telemetry

    def __init__(self):
        self.rng = np.random.default_rng(self.SEED)

    def boot(self):
        print(f"{C.CYAN}{C.BOLD}")
        print(r"   ____  ___    ____  __  ________________   __________________ ")
        print(r"  / __ )/   |  / __ \ \ \/ / ____/ ____/ | / /_  __/ ____/ __ \ ")
        print(r" / __  / /| | / /_/ /  \  / /   / __/ /  |/ / / / / __/ / /_/ / ")
        print(r"/ /_/ / ___ |/ _, _/   / / /___/ /___/ /|  / / / / /___/ _, _/  ")
        print(r"/_____/_/  |_/_/ |_|   /_/\____/_____/_/ |_/ /_/ /_____/_/ |_|  ")
        print(f"{C.RST}")
        txt(f"{C.MAG}PROJECT BARYCENTER{C.RST} :: The Ontology of Zero :: ArgonOS v9 swarm runtime", indent=2)
        txt(f"{C.DIM}python {sys.version.split()[0]} | numpy {np.__version__} | "
            f"float64 eps {np.finfo(np.float64).eps:.2e} | seed 0x{self.SEED:X}{C.RST}", indent=2)
        txt(f"{C.DIM}hypothesis: 0 is an emergent interference / orthogonality / collapse state of non-zero entities{C.RST}",
            indent=2)

    def execute(self) -> int:
        self.boot()
        results = []
        agents = [
            ("A1", lambda: Agent1HardwareEngineer().run()),
            ("A2", lambda: Agent2HypercomplexAlgebraist().run(self.rng)),
            ("A3", lambda: Agent3StatisticalPhysicist().run(self.rng)),
            ("A4", lambda: Agent4LatentSpaceArchitect().run(self.rng)),
        ]
        timings = []
        for tag, fn in agents:
            t0 = time.perf_counter()
            try:
                res = fn()
            except Exception as exc:  # telemetry must survive an agent crash
                res = {"agent": tag, "zero": f"EXCEPTION: {exc!r}", "passed": False}
                print(f"\n  {C.RED}{C.BOLD}[{tag}] AGENT FAULT: {exc!r}{C.RST}")
            timings.append(time.perf_counter() - t0)
            results.append(res)

        banner("LEAD ARCHITECT :: SWARM CONSENSUS REPORT", "All zeros produced by non-zero inputs")
        hdr = f"{'agent':<26} │ {'emergent zero':<50} │ {'t(s)':>6} │ status"
        txt(f"{C.DIM}{hdr}{C.RST}", indent=2)
        txt(f"{C.DIM}{'─' * (len(hdr) + 4)}{C.RST}", indent=2)
        for res, t in zip(results, timings):
            z = res["zero"] if len(res["zero"]) <= 50 else res["zero"][:47] + "..."
            txt(f"{res['agent']:<26} │ {z:<50} │ {t:>6.2f} │ {ok(res['passed'])}", indent=2)

        all_pass = all(r["passed"] for r in results)
        print()
        if all_pass:
            verdict([
                "HARDWARE  : zero is an asserted flag (ZF=1) and a signed bit pattern (±0).",
                "ALGEBRA   : zero is the mutual annihilation of two invertible 16D sedenions.",
                "PHYSICS   : zero is the barycenter of a million maximally active oscillators.",
                "GEOMETRY  : zero is the 90° relation between two fully intact 512-D embeddings.",
                "CONSENSUS : in all four domains, zero was manufactured by non-zero entities.",
            ])
            txt(f"\n  {C.GRN}{C.BOLD}■ PROJECT BARYCENTER: 4/4 EXPERIMENTS VERIFIED — EXIT 0{C.RST}", indent=0)
            return 0
        n_ok = sum(r["passed"] for r in results)
        txt(f"\n  {C.RED}{C.BOLD}■ PROJECT BARYCENTER: {n_ok}/4 EXPERIMENTS VERIFIED — EXIT 1{C.RST}", indent=0)
        return 1


if __name__ == "__main__":
    sys.exit(LeadArchitect().execute())