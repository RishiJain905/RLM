"""The behaviour library for the needle-function tasks (RepoQA's method, plan step 7).

Each behaviour is a small pure function: a code template, two plain-English descriptions, and a reference
implementation that the tests run the generated code against, so a description can never drift from its code.
Behaviours come in groups of near neighbours (sum of squares of the even items, of the odd items, of cubes),
which are each other's decoys: telling them apart means reading the code or running it.

Templates are `str.format` strings. `{a}` and `{b}` are the arguments, `{v}`, `{w}` and `{it}` are local names,
all renamed per seed. Literal braces are doubled.

`TRAIN_GROUPS` feed the training family. `HELDOUT_GROUPS` appear only in the held-out test (H7), with their own
descriptions, so the held-out test asks about behaviours the policy never saw described.
"""

from __future__ import annotations

import collections
import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Behaviour:
    key: str
    group: str
    args: int  # 1 or 2
    template: str
    descriptions: tuple[str, ...]
    reference: Callable[..., Any]
    samples: tuple  # argument tuples the tests run both implementations on


_INTS = ([3, 4, 7, 10, -2, 5], [1, 1, 2, 3, 5, 8, 13], [], [-5, 0, 6, -1, 9, 2], [4])
_WORDS = ("the quick brown fox jumps over the lazy dog", "a bb ccc dd e", "Hello world", "", "same size word")


def _one(*xs):
    return tuple((x,) for x in xs)


TRAIN = [
    # --- sums over a filtered list
    Behaviour("even_square_sum", "filtered_sum", 1,
              "{v} = 0\nfor {it} in {a}:\n    if {it} % 2 == 0:\n        {v} += {it} * {it}\nreturn {v}",
              ("adds up the squares of the even numbers in a list",
               "returns the sum of x squared over every even x in its input"),
              lambda xs: sum(x * x for x in xs if x % 2 == 0), _one(*_INTS)),
    Behaviour("odd_square_sum", "filtered_sum", 1,
              "{v} = 0\nfor {it} in {a}:\n    if {it} % 2 != 0:\n        {v} += {it} * {it}\nreturn {v}",
              ("adds up the squares of the odd numbers in a list",
               "returns the sum of x squared over every odd x in its input"),
              lambda xs: sum(x * x for x in xs if x % 2 != 0), _one(*_INTS)),
    Behaviour("even_cube_sum", "filtered_sum", 1,
              "{v} = 0\nfor {it} in {a}:\n    if {it} % 2 == 0:\n        {v} += {it} ** 3\nreturn {v}",
              ("adds up the cubes of the even numbers in a list",
               "returns the sum of x cubed over every even x in its input"),
              lambda xs: sum(x ** 3 for x in xs if x % 2 == 0), _one(*_INTS)),
    Behaviour("even_count", "filtered_sum", 1,
              "return sum(1 for {it} in {a} if {it} % 2 == 0)",
              ("counts how many numbers in a list are even",
               "returns the number of even values in its input"),
              lambda xs: sum(1 for x in xs if x % 2 == 0), _one(*_INTS)),
    # --- de-duplication
    Behaviour("dedupe_first", "dedupe", 1,
              "{v} = set()\n{w} = []\nfor {it} in {a}:\n    if {it} not in {v}:\n        {v}.add({it})\n"
              "        {w}.append({it})\nreturn {w}",
              ("removes repeated items from a list, keeping each item where it first appears",
               "drops duplicates while preserving the order of first occurrence"),
              lambda xs: list(dict.fromkeys(xs)), _one([3, 1, 3, 2, 1, 5], [], [7, 7, 7], [1, 2, 3])),
    Behaviour("dedupe_last", "dedupe", 1,
              "{v} = {{}}\nfor {it}, {w} in enumerate({a}):\n    {v}[{w}] = {it}\n"
              "return [{w} for {w}, _ in sorted({v}.items(), key=lambda p: p[1])]",
              ("removes repeated items from a list, keeping each item where it last appears",
               "drops duplicates, ordering the survivors by their final occurrence"),
              lambda xs: [x for x, _ in sorted({x: i for i, x in enumerate(xs)}.items(), key=lambda p: p[1])],
              _one([3, 1, 3, 2, 1, 5], [], [7, 7, 7], [1, 2, 3])),
    Behaviour("sorted_unique", "dedupe", 1,
              "return sorted(set({a}))",
              ("returns the distinct items of a list in ascending order",
               "gives the unique values of its input, sorted from smallest to largest"),
              lambda xs: sorted(set(xs)), _one([3, 1, 3, 2, 1, 5], [], [7, 7, 7], [1, 2, 3])),
    Behaviour("singletons", "dedupe", 1,
              "{v} = {{}}\nfor {it} in {a}:\n    {v}[{it}] = {v}.get({it}, 0) + 1\n"
              "return [{it} for {it} in {a} if {v}[{it}] == 1]",
              ("keeps only the items that occur exactly once in a list, in their original order",
               "returns the values that are never repeated in its input"),
              lambda xs: [x for x in xs if xs.count(x) == 1], _one([3, 1, 3, 2, 1, 5], [], [7, 7, 7], [1, 2, 3])),
    # --- rotation
    Behaviour("rotate_left", "rotate", 2,
              "if not {a}:\n    return []\n{v} = {b} % len({a})\nreturn {a}[{v}:] + {a}[:{v}]",
              ("rotates a list to the left by a given number of positions",
               "shifts every element k places towards the front, wrapping the first ones round to the end"),
              lambda xs, k: (xs[k % len(xs):] + xs[:k % len(xs)]) if xs else [],
              (([1, 2, 3, 4, 5], 2), ([1, 2, 3], 4), ([], 3), ([9, 8], 1))),
    Behaviour("rotate_right", "rotate", 2,
              "if not {a}:\n    return []\n{v} = {b} % len({a})\nreturn {a}[len({a}) - {v}:] + {a}[:len({a}) - {v}]",
              ("rotates a list to the right by a given number of positions",
               "shifts every element k places towards the end, wrapping the last ones round to the front"),
              lambda xs, k: (xs[len(xs) - k % len(xs):] + xs[:len(xs) - k % len(xs)]) if xs else [],
              (([1, 2, 3, 4, 5], 2), ([1, 2, 3], 4), ([], 3), ([9, 8], 1))),
    Behaviour("reverse_prefix", "rotate", 2,
              "return list(reversed({a}[:{b}])) + {a}[{b}:]",
              ("reverses the first k items of a list and leaves the rest alone",
               "flips the order of the leading k elements only"),
              lambda xs, k: list(reversed(xs[:k])) + xs[k:],
              (([1, 2, 3, 4, 5], 2), ([1, 2, 3], 4), ([], 3), ([9, 8], 1))),
    # --- running aggregates
    Behaviour("running_max", "running", 1,
              "{w} = []\n{v} = None\nfor {it} in {a}:\n    {v} = {it} if {v} is None or {it} > {v} else {v}\n"
              "    {w}.append({v})\nreturn {w}",
              ("returns the running maximum of a list: each position holds the largest value seen so far",
               "computes the prefix maxima of its input"),
              lambda xs: [max(xs[:i + 1]) for i in range(len(xs))], _one(*_INTS)),
    Behaviour("running_min", "running", 1,
              "{w} = []\n{v} = None\nfor {it} in {a}:\n    {v} = {it} if {v} is None or {it} < {v} else {v}\n"
              "    {w}.append({v})\nreturn {w}",
              ("returns the running minimum of a list: each position holds the smallest value seen so far",
               "computes the prefix minima of its input"),
              lambda xs: [min(xs[:i + 1]) for i in range(len(xs))], _one(*_INTS)),
    Behaviour("running_sum", "running", 1,
              "{w} = []\n{v} = 0\nfor {it} in {a}:\n    {v} += {it}\n    {w}.append({v})\nreturn {w}",
              ("returns the running total of a list: each position holds the sum of everything up to it",
               "computes the prefix sums of its input"),
              lambda xs: [sum(xs[:i + 1]) for i in range(len(xs))], _one(*_INTS)),
    # --- words
    Behaviour("longest_word_first", "words", 1,
              "{v} = ''\nfor {it} in {a}.split():\n    if len({it}) > len({v}):\n        {v} = {it}\nreturn {v}",
              ("finds the longest word in a sentence, preferring the earliest one on a tie",
               "returns the first of the longest whitespace-separated words in a string"),
              lambda s: max(s.split(), key=len, default=""), _one(*_WORDS)),
    Behaviour("longest_word_last", "words", 1,
              "{v} = ''\nfor {it} in {a}.split():\n    if len({it}) >= len({v}):\n        {v} = {it}\nreturn {v}",
              ("finds the longest word in a sentence, preferring the latest one on a tie",
               "returns the last of the longest whitespace-separated words in a string"),
              lambda s: max(reversed(s.split()), key=len, default=""), _one(*_WORDS)),
    Behaviour("shortest_word", "words", 1,
              "{v} = None\nfor {it} in {a}.split():\n    if {v} is None or len({it}) < len({v}):\n        {v} = {it}\n"
              "return {v} or ''",
              ("finds the shortest word in a sentence, preferring the earliest one on a tie",
               "returns the first of the shortest whitespace-separated words in a string"),
              lambda s: min(s.split(), key=len, default=""), _one(*_WORDS)),
    Behaviour("long_word_count", "words", 1,
              "return len([{it} for {it} in {a}.split() if len({it}) > 3])",
              ("counts the words in a sentence that are longer than three characters",
               "returns how many whitespace-separated words have four or more characters"),
              lambda s: sum(len(w) > 3 for w in s.split()), _one(*_WORDS)),
    # --- character classes
    Behaviour("count_vowels", "chars", 1,
              "return sum(1 for {it} in {a}.lower() if {it} in 'aeiou')",
              ("counts the vowels in a string, ignoring case",
               "returns how many of its characters are a, e, i, o or u in either case"),
              lambda s: sum(c in "aeiou" for c in s.lower()), _one(*_WORDS)),
    Behaviour("count_consonants", "chars", 1,
              "return sum(1 for {it} in {a}.lower() if {it}.isalpha() and {it} not in 'aeiou')",
              ("counts the consonants in a string, ignoring case",
               "returns how many of its characters are letters other than a, e, i, o and u"),
              lambda s: sum(c.isalpha() and c not in "aeiou" for c in s.lower()), _one(*_WORDS)),
    Behaviour("count_upper", "chars", 1,
              "return sum(1 for {it} in {a} if {it}.isupper())",
              ("counts the uppercase letters in a string",
               "returns how many capital letters its input contains"),
              lambda s: sum(c.isupper() for c in s), _one(*_WORDS)),
    # --- palindromes
    Behaviour("palindrome_loose", "palindrome", 1,
              "{v} = [{it}.lower() for {it} in {a} if {it}.isalnum()]\nreturn {v} == {v}[::-1]",
              ("checks whether a string reads the same backwards, ignoring case, spaces and punctuation",
               "tests for a palindrome after keeping only letters and digits and lowering the case"),
              lambda s: [c.lower() for c in s if c.isalnum()] == [c.lower() for c in s if c.isalnum()][::-1],
              _one("A man, a plan, a canal: Panama", "Abba", "abc", "Never odd or even", "ab ba")),
    Behaviour("palindrome_exact", "palindrome", 1,
              "return {a} == {a}[::-1]",
              ("checks whether a string is exactly the same when reversed, character for character",
               "tests for a palindrome without ignoring case, spaces or punctuation"),
              lambda s: s == s[::-1], _one("A man, a plan, a canal: Panama", "Abba", "abc", "abba", "ab ba")),
    Behaviour("palindrome_case", "palindrome", 1,
              "{v} = {a}.lower()\nreturn {v} == {v}[::-1]",
              ("checks whether a string reads the same backwards when case is ignored, but spaces still count",
               "tests for a palindrome after lowering the case only"),
              lambda s: s.lower() == s.lower()[::-1], _one("A man, a plan, a canal: Panama", "Abba", "abc", "ab ba")),
    # --- chunking
    Behaviour("chunks", "chunk", 2,
              "return [{a}[{it}:{it} + {b}] for {it} in range(0, len({a}), {b})]",
              ("splits a list into consecutive pieces of size n, the last piece possibly shorter",
               "cuts its input into back-to-back groups of n items, keeping a short final group"),
              lambda xs, n: [xs[i:i + n] for i in range(0, len(xs), n)],
              (([1, 2, 3, 4, 5], 2), ([1, 2, 3], 3), ([], 2), ([1, 2, 3, 4, 5, 6, 7], 3))),
    Behaviour("full_chunks", "chunk", 2,
              "return [{a}[{it}:{it} + {b}] for {it} in range(0, len({a}) - {b} + 1, {b})]",
              ("splits a list into consecutive pieces of size n, dropping a shorter leftover piece",
               "cuts its input into back-to-back groups of exactly n items, discarding any remainder"),
              lambda xs, n: [xs[i:i + n] for i in range(0, len(xs) - n + 1, n)],
              (([1, 2, 3, 4, 5], 2), ([1, 2, 3], 3), ([], 2), ([1, 2, 3, 4, 5, 6, 7], 3))),
    Behaviour("windows", "chunk", 2,
              "return [{a}[{it}:{it} + {b}] for {it} in range(len({a}) - {b} + 1)]",
              ("returns every overlapping window of n consecutive items in a list",
               "slides a window of width n along its input one step at a time and collects each window"),
              lambda xs, n: [xs[i:i + n] for i in range(len(xs) - n + 1)],
              (([1, 2, 3, 4, 5], 2), ([1, 2, 3], 3), ([], 2), ([1, 2, 3, 4, 5, 6, 7], 3))),
    # --- centre and spread
    Behaviour("median", "centre", 1,
              "{v} = sorted({a})\nif not {v}:\n    return None\n{w} = len({v}) // 2\n"
              "return {v}[{w}] if len({v}) % 2 else ({v}[{w} - 1] + {v}[{w}]) / 2",
              ("returns the median of a list of numbers, averaging the two middle values when the count is even",
               "computes the middle value of its sorted input"),
              lambda xs: statistics.median(xs) if xs else None, _one(*_INTS)),
    Behaviour("mean", "centre", 1,
              "return sum({a}) / len({a}) if {a} else None",
              ("returns the arithmetic mean of a list of numbers",
               "computes the average of its input"),
              lambda xs: sum(xs) / len(xs) if xs else None, _one(*_INTS)),
    Behaviour("value_range", "centre", 1,
              "return max({a}) - min({a}) if {a} else None",
              ("returns the difference between the largest and smallest numbers in a list",
               "computes the spread, maximum minus minimum, of its input"),
              lambda xs: max(xs) - min(xs) if xs else None, _one(*_INTS)),
    # --- searching for negatives
    Behaviour("first_negative", "search", 1,
              "for {it}, {v} in enumerate({a}):\n    if {v} < 0:\n        return {it}\nreturn -1",
              ("returns the index of the first negative number in a list, or -1 if there is none",
               "finds the position of the earliest value below zero"),
              lambda xs: next((i for i, x in enumerate(xs) if x < 0), -1), _one(*_INTS)),
    Behaviour("last_negative", "search", 1,
              "{w} = -1\nfor {it}, {v} in enumerate({a}):\n    if {v} < 0:\n        {w} = {it}\nreturn {w}",
              ("returns the index of the last negative number in a list, or -1 if there is none",
               "finds the position of the latest value below zero"),
              lambda xs: max((i for i, x in enumerate(xs) if x < 0), default=-1), _one(*_INTS)),
    Behaviour("first_zero", "search", 1,
              "for {it}, {v} in enumerate({a}):\n    if {v} == 0:\n        return {it}\nreturn -1",
              ("returns the index of the first zero in a list, or -1 if there is none",
               "finds the position of the earliest value equal to zero"),
              lambda xs: next((i for i, x in enumerate(xs) if x == 0), -1), _one(*_INTS)),
    # --- letter shifting
    Behaviour("shift_forward", "cipher", 2,
              "{w} = []\nfor {it} in {a}:\n    if {it}.islower():\n        {w}.append(chr((ord({it}) - 97 + {b}) % 26 + 97))\n"
              "    elif {it}.isupper():\n        {w}.append(chr((ord({it}) - 65 + {b}) % 26 + 65))\n    else:\n"
              "        {w}.append({it})\nreturn ''.join({w})",
              ("shifts every letter of a string forward through the alphabet by n places, wrapping z round to a",
               "applies a Caesar shift of +n to its input, leaving non-letters unchanged"),
              lambda s, n: "".join(chr((ord(c) - 97 + n) % 26 + 97) if c.islower() else chr((ord(c) - 65 + n) % 26 + 65)
                                   if c.isupper() else c for c in s),
              (("Hello, World", 3), ("xyz", 2), ("", 5), ("abc", 26))),
    Behaviour("shift_back", "cipher", 2,
              "{w} = []\nfor {it} in {a}:\n    if {it}.islower():\n        {w}.append(chr((ord({it}) - 97 - {b}) % 26 + 97))\n"
              "    elif {it}.isupper():\n        {w}.append(chr((ord({it}) - 65 - {b}) % 26 + 65))\n    else:\n"
              "        {w}.append({it})\nreturn ''.join({w})",
              ("shifts every letter of a string backwards through the alphabet by n places, wrapping a round to z",
               "applies a Caesar shift of -n to its input, leaving non-letters unchanged"),
              lambda s, n: "".join(chr((ord(c) - 97 - n) % 26 + 97) if c.islower() else chr((ord(c) - 65 - n) % 26 + 65)
                                   if c.isupper() else c for c in s),
              (("Hello, World", 3), ("xyz", 2), ("", 5), ("abc", 26))),
    Behaviour("reverse_words", "cipher", 1,
              "return ' '.join({it}[::-1] for {it} in {a}.split(' '))",
              ("reverses the letters of each word in a string but keeps the words in place",
               "spells every space-separated word backwards without reordering the words"),
              lambda s: " ".join(w[::-1] for w in s.split(" ")), _one(*_WORDS)),
    # --- nesting
    Behaviour("flatten_once", "flatten", 1,
              "{w} = []\nfor {it} in {a}:\n    if isinstance({it}, list):\n        {w}.extend({it})\n    else:\n"
              "        {w}.append({it})\nreturn {w}",
              ("flattens a list by one level: items that are lists are spliced in, deeper lists are left as they are",
               "removes a single layer of nesting from its input"),
              lambda xs: [y for x in xs for y in (x if isinstance(x, list) else [x])],
              _one([1, [2, [3, 4]], 5], [], [[1], [2, 3]], [1, 2])),
    Behaviour("flatten_deep", "flatten", 1,
              "{w} = []\n{v} = list(reversed({a}))\nwhile {v}:\n    {it} = {v}.pop()\n    if isinstance({it}, list):\n"
              "        {v}.extend(reversed({it}))\n    else:\n        {w}.append({it})\nreturn {w}",
              ("flattens a nested list completely, however deep it goes, keeping the left-to-right order",
               "removes every layer of nesting from its input"),
              lambda xs: _deep(xs), _one([1, [2, [3, 4]], 5], [], [[1], [2, 3]], [1, 2])),
    # --- key/value parsing
    Behaviour("parse_semicolon_pairs", "parse", 1,
              "{w} = {{}}\nfor {it} in {a}.split(';'):\n    if '=' in {it}:\n        {v}, _, {it} = {it}.partition('=')\n"
              "        {w}[{v}.strip()] = {it}.strip()\nreturn {w}",
              ("parses a string like 'a=1; b=2' into a dictionary of keys and values",
               "turns semicolon-separated key=value pairs into a dict of stripped strings"),
              lambda s: {p.partition("=")[0].strip(): p.partition("=")[2].strip() for p in s.split(";") if "=" in p},
              _one("a=1; b=2", "x = y", "", "k=v;bad;z=3")),
    Behaviour("parse_comma_colon", "parse", 1,
              "{w} = {{}}\nfor {it} in {a}.split(','):\n    if ':' in {it}:\n        {v}, _, {it} = {it}.partition(':')\n"
              "        {w}[{v}.strip()] = {it}.strip()\nreturn {w}",
              ("parses a string like 'a:1, b:2' into a dictionary of keys and values",
               "turns comma-separated key:value pairs into a dict of stripped strings"),
              lambda s: {p.partition(":")[0].strip(): p.partition(":")[2].strip() for p in s.split(",") if ":" in p},
              _one("a:1, b:2", "x : y", "", "k:v,bad,z:3")),
    # --- digits
    Behaviour("digit_sum", "digits", 1,
              "return sum(int({it}) for {it} in str(abs({a})))",
              ("adds up the decimal digits of an integer, ignoring its sign",
               "returns the sum of the digits of its input"),
              lambda n: sum(int(c) for c in str(abs(n))), _one(0, 7, 1234, -905, 99999)),
    Behaviour("digit_product", "digits", 1,
              "{v} = 1\nfor {it} in str(abs({a})):\n    {v} *= int({it})\nreturn {v}",
              ("multiplies together the decimal digits of an integer, ignoring its sign",
               "returns the product of the digits of its input"),
              lambda n: math.prod(int(c) for c in str(abs(n))), _one(0, 7, 1234, -905, 99999)),
    Behaviour("digit_root", "digits", 1,
              "{v} = abs({a})\nwhile {v} >= 10:\n    {v} = sum(int({it}) for {it} in str({v}))\nreturn {v}",
              ("repeatedly sums the digits of an integer until a single digit is left",
               "returns the digital root of its input"),
              lambda n: _root(abs(n)), _one(0, 7, 1234, -905, 99999)),
    # --- frequencies
    Behaviour("most_common_word", "frequency", 1,
              "{v} = {{}}\nfor {it} in {a}.lower().split():\n    {v}[{it}] = {v}.get({it}, 0) + 1\n"
              "return min({v}, key=lambda {w}: (-{v}[{w}], {w})) if {v} else None",
              ("returns the most frequent word in a text, lowercased, breaking ties alphabetically",
               "finds the word that occurs most often, choosing the alphabetically first on a tie"),
              lambda s: min(collections.Counter(s.lower().split()).items(), key=lambda p: (-p[1], p[0]))[0]
              if s.split() else None, _one(*_WORDS, "b a b a c")),
    Behaviour("least_common_word", "frequency", 1,
              "{v} = {{}}\nfor {it} in {a}.lower().split():\n    {v}[{it}] = {v}.get({it}, 0) + 1\n"
              "return min({v}, key=lambda {w}: ({v}[{w}], {w})) if {v} else None",
              ("returns the least frequent word in a text, lowercased, breaking ties alphabetically",
               "finds the word that occurs least often, choosing the alphabetically first on a tie"),
              lambda s: min(collections.Counter(s.lower().split()).items(), key=lambda p: (p[1], p[0]))[0]
              if s.split() else None, _one(*_WORDS, "b a b a c")),
    Behaviour("distinct_words", "frequency", 1,
              "return len(set({a}.lower().split()))",
              ("counts the distinct words in a text, ignoring case",
               "returns how many different lowercased words its input contains"),
              lambda s: len(set(s.lower().split())), _one(*_WORDS, "b a b a c")),
]

HELDOUT = [
    Behaviour("title_case", "casing", 1,
              "return ' '.join({it}[:1].upper() + {it}[1:].lower() for {it} in {a}.split(' '))",
              ("capitalises the first letter of every word and lowercases the rest of each word",),
              lambda s: " ".join(w[:1].upper() + w[1:].lower() for w in s.split(" ")), _one(*_WORDS, "hELLO tHERE")),
    Behaviour("swap_case", "casing", 1,
              "return ''.join({it}.lower() if {it}.isupper() else {it}.upper() for {it} in {a})",
              ("turns every uppercase letter into lowercase and every lowercase letter into uppercase",),
              lambda s: s.swapcase(), _one(*_WORDS, "hELLO tHERE")),
    Behaviour("sentence_case", "casing", 1,
              "return {a}[:1].upper() + {a}[1:].lower()",
              ("uppercases only the very first character of the text and lowercases everything after it",),
              lambda s: s[:1].upper() + s[1:].lower(), _one(*_WORDS, "hELLO tHERE")),
    Behaviour("pair_count", "pairs", 2,
              "{v} = 0\nfor {it} in range(len({a})):\n    for {w} in range({it} + 1, len({a})):\n"
              "        if {a}[{it}] + {a}[{w}] == {b}:\n            {v} += 1\nreturn {v}",
              ("tells you how many unordered pairs of positions in the sequence hold values adding up to the target",),
              lambda xs, t: sum(xs[i] + xs[j] == t for i in range(len(xs)) for j in range(i + 1, len(xs))),
              (([1, 2, 3, 4, 5], 5), ([2, 2, 2], 4), ([], 1), ([0, -1, 1], 0))),
    Behaviour("pair_diff_count", "pairs", 2,
              "{v} = 0\nfor {it} in range(len({a})):\n    for {w} in range({it} + 1, len({a})):\n"
              "        if abs({a}[{it}] - {a}[{w}]) == {b}:\n            {v} += 1\nreturn {v}",
              ("tells you how many unordered pairs of positions hold values whose absolute difference is the target",),
              lambda xs, t: sum(abs(xs[i] - xs[j]) == t for i in range(len(xs)) for j in range(i + 1, len(xs))),
              (([1, 2, 3, 4, 5], 2), ([2, 2, 2], 0), ([], 1), ([0, -1, 1], 1))),
    Behaviour("set_bits", "bits", 1,
              "{v} = 0\n{w} = abs({a})\nwhile {w}:\n    {v} += {w} & 1\n    {w} >>= 1\nreturn {v}",
              ("reports how many 1s appear in the binary form of the absolute value of an integer",),
              lambda n: bin(abs(n)).count("1"), _one(0, 1, 7, 8, -12, 255)),
    Behaviour("trailing_zeros", "bits", 1,
              "{w} = abs({a})\nif {w} == 0:\n    return 0\n{v} = 0\nwhile {w} % 2 == 0:\n    {v} += 1\n    {w} //= 2\nreturn {v}",
              ("reports how many 0s come after the last 1 in the binary form of an integer, with zero giving 0",),
              lambda n: 0 if n == 0 else (abs(n) & -abs(n)).bit_length() - 1, _one(0, 1, 7, 8, -12, 256)),
    Behaviour("second_largest", "ranking", 1,
              "{v} = sorted(set({a}), reverse=True)\nreturn {v}[1] if len({v}) > 1 else None",
              ("gives the runner-up among the distinct values, the biggest one below the maximum",),
              lambda xs: sorted(set(xs), reverse=True)[1] if len(set(xs)) > 1 else None, _one(*_INTS, [5, 5, 4])),
    Behaviour("second_smallest", "ranking", 1,
              "{v} = sorted(set({a}))\nreturn {v}[1] if len({v}) > 1 else None",
              ("gives the distinct value sitting just above the minimum",),
              lambda xs: sorted(set(xs))[1] if len(set(xs)) > 1 else None, _one(*_INTS, [5, 5, 4])),
    Behaviour("balanced", "brackets", 1,
              "{v} = 0\nfor {it} in {a}:\n    if {it} == '(':\n        {v} += 1\n    elif {it} == ')':\n        {v} -= 1\n"
              "        if {v} < 0:\n            return False\nreturn {v} == 0",
              ("decides whether every opening round bracket in the text is closed in the right order",),
              lambda s: _balanced(s), _one("(a(b)c)", "(()", "())(", "", "x")),
    Behaviour("max_depth", "brackets", 1,
              "{v} = 0\n{w} = 0\nfor {it} in {a}:\n    if {it} == '(':\n        {v} += 1\n        {w} = max({w}, {v})\n"
              "    elif {it} == ')':\n        {v} -= 1\nreturn {w}",
              ("reports how deeply round brackets are nested at the deepest point of the text",),
              lambda s: _depth(s), _one("(a(b)c)", "(()", "())(", "", "x")),
]


def _deep(xs):
    out = []
    for x in xs:
        out.extend(_deep(x) if isinstance(x, list) else [x])
    return out


def _root(n):
    while n >= 10:
        n = sum(int(c) for c in str(n))
    return n


def _balanced(s):
    depth = 0
    for c in s:
        depth += (c == "(") - (c == ")")
        if depth < 0:
            return False
    return depth == 0


def _depth(s):
    depth = best = 0
    for c in s:
        depth += (c == "(") - (c == ")")
        best = max(best, depth)
    return best


def groups(behaviours: list[Behaviour]) -> dict[str, list[Behaviour]]:
    out: dict[str, list[Behaviour]] = collections.defaultdict(list)
    for b in behaviours:
        out[b.group].append(b)
    return dict(out)


TRAIN_GROUPS = groups(TRAIN)
HELDOUT_GROUPS = groups(HELDOUT)



def render(behaviour: Behaviour, names: dict[str, str]) -> str:
    """The function body, indented one level, with the per-seed local names filled in."""
    body = behaviour.template.format(**names)
    return "\n".join("    " + line if line else "" for line in body.splitlines())
