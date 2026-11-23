# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Internal module for copying ``TTrees`` between ROOT files without decompressing
or re-interpreting their ``TBaskets``.

This module implements the ``TTree`` part of
:ref:`uproot.writing.writable.WritableDirectory.copy_from`. For every selected
``TTree`` it

1. *plans* the copy by reading the source ``TTree``/``TBranch``/``TLeaf``
   metadata with the ordinary reading models
   (:doc:`uproot.models.TTree`, :doc:`uproot.models.TBranch`,
   :doc:`uproot.models.TLeaf`), checking that the tree has a layout that the
   cascading writer (:doc:`uproot.writing._cascadetree`) can faithfully
   rebuild, and checking that every ``TBasket`` reference is inside the source
   file — all before anything is written to the destination, then

2. *executes* the copy by moving every ``TBasket``'s bytes verbatim (only the
   ``fSeekKey``/``fSeekPdir`` fields of each basket's ``TKey`` are adapted to
   the destination) and writing one fresh set of ``TTree`` metadata whose
   ``fBasketSeek``/``fBasketEntry``/``fBasketBytes`` locator tables and
   cumulative byte counters (``fTotBytes``/``fZipBytes``) are re-expressed in
   terms of the destination file positions.

The compressed data region of every basket (all bytes after the basket's
``fKeylen``) is copied byte-for-byte: decompression and recompression are never
invoked, and neither is any array interpretation. Event (entry) boundaries,
branch types, names, titles, and counter relationships are preserved by
rebuilding the metadata from the source metadata rather than from arrays.

Basket copy schedule
--------------------

Two schedules were considered:

* *source-position order*: sort all baskets of a tree by their position in the
  source file and copy them in that order. This minimizes seek distance on the
  source (one monotonic pass) and allows adjacent baskets to be merged into
  fewer read requests, but it scatters each branch's baskets across the
  destination exactly as they happen to lie in the source, and it fills the
  ``fBasketSeek`` locator tables out of basket-index order.

* *branch-major order* (chosen): copy all baskets of one branch in basket-index
  order before moving to the next branch. The locator tables are filled
  strictly in basket-index order (the metadata is written exactly once, after
  all payloads are in place), the destination receives each branch's baskets as
  one contiguous run — the same clustering ROOT itself produces with
  ``TTree::FlushBaskets`` and the layout that favors the dominant access
  pattern of reading a few branches in full — and the destination write
  positions are monotonic. The number of source requests is identical for both
  schedules (it is bounded by the payload size, not the schedule); only the
  order of the requests differs. Since the first version of this feature
  targets local files, where repositioning the read cursor is cheap,
  branch-major order is preferred for its simpler locator-table maintenance
  and better destination locality.

I/O bounds
----------

While moving payload bytes, every read request to the source and every write
request to the destination is at most 1 MiB (``_MAX_COPY_REQUEST``), and the
loop holds at most two such buffers at any time (one chunk as read and, for
the first chunk of a basket, one patched copy), so the total resident payload
memory stays below 4 MiB regardless of basket or tree size. ``TTree``
metadata and the basket locator tables are not payload and are not charged
against this budget.

Failure semantics
-----------------

All validation (types, required metadata, and source range boundaries) happens
in the planning phase, before the destination file is modified. Unsupported
layouts raise ``NotImplementedError`` and malformed source data raise
``ValueError``; both name the offending tree and branch. Runtime I/O errors
from the source or destination propagate unchanged.

Baskets are copied before the tree's metadata and directory key are written
(referents before references), so a tree whose copy is interrupted never
appears as a readable key in the destination directory. Objects that were
completely copied before the failure remain readable, and pre-existing cycles
are never modified. On failure, this tree's not-yet-referenced allocations are
returned to the in-memory free-segment list on a best-effort basis; cleanup
errors are never allowed to mask the original exception.
"""
from __future__ import annotations

import contextlib
import os
import re
import struct

import numpy

import uproot._util
import uproot.compression
import uproot.const
import uproot.reading
import uproot.source.chunk
import uproot.writing._cascade
import uproot.writing._cascadetree

#: Maximum number of bytes in a single source read or destination write
#: request while moving basket payloads (1 MiB).
MAX_COPY_REQUEST = 1024 * 1024

#: Maximum number of payload bytes held in memory at once (4 MiB). The copy
#: loop uses at most two MAX_COPY_REQUEST-sized buffers at any time.
MAX_COPY_BUFFER = 4 * 1024 * 1024

_basket_key_head = struct.Struct(">ihiIhh")  # through fCycle (18 bytes)
_basket_key_head_big = struct.Struct(">qq")  # fSeekKey, fSeekPdir (big)
_basket_key_head_small = struct.Struct(">ii")  # fSeekKey, fSeekPdir (small)

# Leaf class name -> (signed/neutral dtype, unsigned dtype or None)
_leaf_class_dtypes = {
    "TLeafO": (numpy.dtype("bool"), None),
    "TLeafB": (numpy.dtype(">i1"), numpy.dtype(">u1")),
    "TLeafS": (numpy.dtype(">i2"), numpy.dtype(">u2")),
    "TLeafI": (numpy.dtype(">i4"), numpy.dtype(">u4")),
    "TLeafL": (numpy.dtype(">i8"), numpy.dtype(">u8")),
    "TLeafF": (numpy.dtype(">f4"), None),
    "TLeafD": (numpy.dtype(">f8"), None),
}

_dim_groups = re.compile(r"\[([^\[\]]*)\]")


def _local_file_path(path):
    """
    Interprets ``path`` as a local filesystem path, returning None if it is
    clearly not one (e.g. a remote URL) or is not known (e.g. a file-like
    object).
    """
    if path is None:
        return None
    path = str(path)
    path = path.removeprefix("file://")
    if "://" in path:
        return None
    return path


def check_source_is_not_destination(source, dest_file):
    """
    Raises a ``ValueError`` if ``source`` (a
    :doc:`uproot.reading.ReadOnlyDirectory`) and ``dest_file`` (a
    :doc:`uproot.writing.writable.WritableFile`) refer to the same local file,
    including path aliases such as symbolic links, hard links, and ``.``/``..``
    components. If either path is not a known local path (remote URLs,
    file-like objects), the check cannot be performed and returns silently.
    """
    source_path = _local_file_path(source.file_path)
    dest_path = _local_file_path(dest_file.sink.file_path)
    if source_path is None or dest_path is None:
        return
    try:
        same = os.path.samefile(source_path, dest_path)
    except OSError:
        same = os.path.realpath(source_path) == os.path.realpath(dest_path)
    if same:
        raise ValueError(
            "copy_from cannot copy from a file onto itself: "
            f"{source_path!r} and {dest_path!r} refer to the same file"
        )


def tree_copy_streamers():
    """
    Returns a list of :doc:`uproot.writing._cascade.RawStreamerInfo` for the
    class versions the cascading TTree writer produces (the same set that
    :ref:`uproot.writing.writable.WritableDirectory.mktree` installs).
    """
    import uproot.models.TBranch
    import uproot.models.TLeaf
    import uproot.models.TTree

    streamers = []
    seen = set()
    for model in (
        uproot.models.TLeaf.Model_TLeafB_v1,
        uproot.models.TLeaf.Model_TLeafS_v1,
        uproot.models.TLeaf.Model_TLeafI_v1,
        uproot.models.TLeaf.Model_TLeafL_v1,
        uproot.models.TLeaf.Model_TLeafF_v1,
        uproot.models.TLeaf.Model_TLeafD_v1,
        uproot.models.TLeaf.Model_TLeafC_v1,
        uproot.models.TLeaf.Model_TLeafO_v1,
        uproot.models.TBranch.Model_TBranch_v13,
        uproot.models.TTree.Model_TTree_v20,
    ):
        for rawstreamer in model.class_rawstreamers:
            classname_version = rawstreamer[-2], rawstreamer[-1]
            if classname_version not in seen:
                seen.add(classname_version)
                streamers.append(
                    uproot.writing._cascade.RawStreamerInfo(*rawstreamer)
                )
    return streamers


def _member(obj, name, default=None):
    """
    Reads a member, returning ``default`` if it is not present (e.g. an older
    class version) or is None.
    """
    out = obj.member(name, none_if_missing=True)
    return default if out is None else out


def _root_string(buf, pos, where):
    """
    Reads one ROOT-serialized string from ``buf`` starting at ``pos``,
    returning ``(bytes, next_position)``.
    """
    if pos + 1 > len(buf):
        raise ValueError(f"truncated string in a {where}")
    (length,) = struct.unpack(">B", buf[pos : pos + 1])
    pos += 1
    if length == 255:
        if pos + 4 > len(buf):
            raise ValueError(f"truncated string in a {where}")
        (length,) = struct.unpack(">I", buf[pos : pos + 4])
        pos += 4
    if pos + length > len(buf):
        raise ValueError(f"truncated string in a {where}")
    return buf[pos : pos + length], pos + length


class BasketHead:
    """
    The validated header fields of one source ``TBasket``.
    """

    __slots__ = ("big", "fKeylen", "fObjlen")

    def __init__(self, big, fObjlen, fKeylen):
        self.big = big
        self.fObjlen = fObjlen
        self.fKeylen = fKeylen


def _parse_basket_head(head, seek, num_bytes, branch_name, old_name, file_path):
    """
    Parses and validates the fixed part of a source ``TBasket``'s TKey (and
    its class and object name strings). ``head`` must contain at least the
    first ``34 + len("TBasket") + len(branch_name) + 4`` bytes of the basket.

    Raises ``ValueError`` if the bytes are not a self-consistent TBasket key
    for ``branch_name`` at file position ``seek``.
    """
    where = f"basket of branch {branch_name!r} in TTree {old_name!r} in file {file_path}"
    if len(head) < 18:
        raise ValueError(f"could not read the TKey header of a {where}")

    (
        fNbytes,
        fVersion,
        fObjlen,
        _fDatime,
        fKeylen,
        _fCycle,
    ) = _basket_key_head.unpack(head[:18])

    if fVersion >= 1000:
        big = True
        key_version = fVersion - 1000
        strings_at = 34
    else:
        big = False
        key_version = fVersion
        strings_at = 26

    if len(head) < strings_at + 8:
        raise ValueError(f"could not read the TKey header of a {where}")

    if big:
        fSeekKey = _basket_key_head_big.unpack(head[18:34])[0]
    else:
        fSeekKey = _basket_key_head_small.unpack(head[18:26])[0]

    if key_version != 4:
        raise ValueError(
            f"unexpected TKey version {fVersion} in a {where} (expected 4 or 1004)"
        )
    if fSeekKey != seek:
        raise ValueError(
            f"TKey fSeekKey is {fSeekKey} but the basket is located at {seek} "
            f"in a {where}"
        )
    if fNbytes != num_bytes:
        raise ValueError(
            f"TKey fNbytes is {fNbytes} but the TBranch fBasketBytes is "
            f"{num_bytes} in a {where}"
        )
    if fObjlen < 0 or fKeylen < strings_at or fKeylen > fNbytes:
        raise ValueError(
            f"inconsistent TKey sizes (fObjlen={fObjlen}, fKeylen={fKeylen}, "
            f"fNbytes={fNbytes}) in a {where}"
        )

    classname, at = _root_string(head, strings_at, where)
    if classname != b"TBasket":
        raise ValueError(
            f"expected a TBasket key but found class name {classname!r} in a {where}"
        )
    name, at = _root_string(head, at, where)
    if name != branch_name.encode(errors="surrogateescape"):
        raise ValueError(
            f"TBasket key has name {name!r}, expected {branch_name.encode(errors='surrogateescape')!r}, "
            f"in a {where}"
        )
    if fKeylen < at + 20:
        # at least a (possibly empty) title string, the TBasket header, and
        # one more byte must fit inside fKeylen
        raise ValueError(f"TKey fKeylen is too small ({fKeylen}) in a {where}")

    return BasketHead(big, fObjlen, fKeylen)


class _BranchCopyPlan:
    __slots__ = (
        "basket_big",
        "basket_bytes",
        "basket_entry",
        "basket_seek",
        "fBasketSize",
        "fCompress",
        "fEntryOffsetLen",
        "fFirstEntry",
        "fOffset",
        "fSplitLevel",
        "fTotBytes",
        "fZipBytes",
        "is_counter",
        "name",
        "tleaf_maximum",
        "tleaf_minimum",
    )

    def __init__(self, name):
        self.name = name
        self.is_counter = False


class _TreeCopyPlan:
    __slots__ = (
        "branch_types",
        "branches",
        "capacity",
        "counter_names",
        "file_path",
        "has_small_keys",
        "metadata",
        "metadata_bytes_estimate",
        "num_baskets",
        "num_entries",
        "source",
        "total_payload_bytes",
        "tree_title",
    )


def _parse_branch_title(title, branch_name, old_name, file_path):
    """
    Splits a TBranch title of the form ``name[dim1][dim2].../L`` into its
    dimension specifiers and type letter. Raises ``NotImplementedError`` if
    the title does not have the leaflist shape that the cascading writer
    reproduces.
    """
    where = f"branch {branch_name!r} of TTree {old_name!r} in file {file_path}"
    if not title.startswith(branch_name):
        raise NotImplementedError(
            f"copy_from cannot copy a TBranch whose title {title!r} does not "
            f"start with the branch name: {where}"
        )
    rest = title[len(branch_name) :]
    dims_part, separator, letter = rest.rpartition("/")
    if not separator or len(letter) != 1 or not letter.isalpha():
        raise NotImplementedError(
            f"copy_from cannot copy a TBranch whose title {title!r} is not a "
            f"leaflist descriptor ('name[dims]/letter'): {where}"
        )
    groups = _dim_groups.findall(dims_part)
    if "".join(f"[{group}]" for group in groups) != dims_part:
        raise NotImplementedError(
            f"copy_from cannot copy a TBranch whose title {title!r} has an "
            f"unrecognized dimension specification: {where}"
        )
    for group in groups:
        if group.isdigit() and str(int(group)) != group:
            raise NotImplementedError(
                f"copy_from cannot copy a TBranch whose title {title!r} has a "
                f"non-canonical dimension {group!r}: {where}"
            )
    return groups, letter


def plan_tree_copy(source, old_name, old_key):
    """
    Reads the metadata of the ``TTree`` named ``old_name`` from ``source`` (a
    :doc:`uproot.reading.ReadOnlyDirectory`) and returns a
    :class:`_TreeCopyPlan` describing exactly how to copy it, or raises

    * ``NotImplementedError`` if the tree is valid but has a layout that this
      version cannot faithfully rebuild (object branches, string branches,
      nested/split branches, embedded baskets, shared counters, non-uniform
      basket counts, cluster ranges, aliases/friends, ...), or
    * ``ValueError`` if the tree's metadata is internally inconsistent or a
      basket reference points outside the source file.

    This function only reads from the source; it never writes to the
    destination.
    """
    import uproot.behaviors.TBranch
    import uproot.behaviors.TTree

    file = source.file
    file_path = file.file_path
    tree = old_key.get()
    if not isinstance(tree, uproot.behaviors.TTree.TTree):
        raise NotImplementedError(
            f"copy_from expected a TTree at {old_name!r} but got "
            f"{type(tree).__name__} in file {file_path}"
        )

    def unsupported(why):
        return NotImplementedError(
            f"copy_from cannot copy TTree {old_name!r} in file {file_path}: {why}"
        )

    top_branches = list(tree.branches)
    if len(top_branches) == 0:
        raise unsupported("the TTree has no branches")

    tree_entries = _member(tree, "fEntries", 0)

    # Tree-level features the cascading writer cannot reproduce.
    if _member(tree, "fNClusterRange", 0) != 0:
        raise unsupported("the TTree has cluster ranges (fNClusterRange != 0)")
    for member_name in ("fAliases", "fFriends", "fUserInfo", "fTreeIndex", "fBranchRef"):
        if tree.member(member_name, none_if_missing=True) is not None:
            raise unsupported(f"the TTree has a non-empty {member_name}")
    iofeatures = tree.member("fIOFeatures", none_if_missing=True)
    if iofeatures is not None and _member(iofeatures, "fIOBits", 0) != 0:
        raise unsupported("the TTree has non-zero fIOFeatures.fIOBits")

    # First pass: classify branches and collect counter references.
    counter_referrers = {}  # counter branch name -> list of jagged branch names
    branch_records = []  # (branch, leaf, dtype, groups, letter, counter_branch or None)
    for branch in top_branches:
        bname = branch.member("fName")
        if branch.classname != "TBranch":
            raise unsupported(
                f"branch {bname!r} has class {branch.classname} (only flat "
                "TBranch branches of basic types can be copied)"
            )
        if len(branch.branches) != 0:
            raise unsupported(
                f"branch {bname!r} has sub-branches (split/object layouts "
                "cannot be copied)"
            )
        leaves = branch.member("fLeaves")
        if len(leaves) != 1:
            raise unsupported(
                f"branch {bname!r} has {len(leaves)} leaves (only single-leaf "
                "branches can be copied)"
            )
        leaf = leaves[0]
        leaf_class = leaf.classname
        if leaf_class not in _leaf_class_dtypes:
            raise unsupported(
                f"branch {bname!r} has a {leaf_class} leaf (only numeric "
                "scalar, fixed-size array, and single-level variable-length "
                "numeric branches can be copied)"
            )
        if leaf.member("fName") != bname:
            raise unsupported(
                f"the leaf of branch {bname!r} has a different name "
                f"({leaf.member('fName')!r})"
            )
        signed_dtype, unsigned_dtype = _leaf_class_dtypes[leaf_class]
        if leaf.member("fIsUnsigned") and unsigned_dtype is not None:
            dtype = unsigned_dtype
        else:
            dtype = signed_dtype

        groups, letter = _parse_branch_title(
            branch.member("fTitle"), bname, old_name, file_path
        )
        expected_letter = uproot.writing._cascadetree._dtype_to_char[dtype]
        if letter != expected_letter:
            raise ValueError(
                f"branch {bname!r} of TTree {old_name!r} in file {file_path} "
                f"has type letter {letter!r} in its title but a {leaf_class} "
                f"leaf (expected letter {expected_letter!r})"
            )

        leaf_count = leaf.member("fLeafCount", none_if_missing=True)
        counter_branch = None
        if leaf_count is not None:
            parent = getattr(leaf_count, "parent", None)
            if (
                parent is None
                or not isinstance(parent, uproot.behaviors.TBranch.TBranch)
                or parent.tree is not tree
                or not parent.top_level
            ):
                raise unsupported(
                    f"the count leaf of branch {bname!r} does not belong to a "
                    "top-level branch of the same TTree"
                )
            counter_branch = parent
            counter_referrers.setdefault(counter_branch.member("fName"), []).append(
                bname
            )

        branch_records.append(
            (branch, leaf, dtype, groups, letter, leaf_count, counter_branch)
        )

    for counter_name, referrers in counter_referrers.items():
        if len(referrers) != 1:
            raise unsupported(
                f"counter branch {counter_name!r} is shared by several "
                f"branches ({', '.join(referrers)})"
            )

    # Second pass: per-branch validation and basket-locator checks.
    num_baskets = None
    plans = []
    branch_types = {}
    counter_names = {}
    total_payload_bytes = 0
    has_small_keys = False
    sum_tot_bytes = 0
    sum_zip_bytes = 0
    max_basket_capacity = 0

    for branch, leaf, dtype, groups, _letter, leaf_count, _counter_branch in branch_records:
        bname = branch.member("fName")
        is_counter = bname in counter_referrers

        where = f"branch {bname!r} of TTree {old_name!r} in file {file_path}"

        # Dimension structure: an optional counter group followed by numeric groups.
        if leaf_count is None:
            counter_dim = None
            numeric_groups = groups
        else:
            if (
                not groups
                or groups[0].isdigit()
                or groups[0] != leaf_count.member("fName")
            ):
                raise ValueError(
                    f"the title of {where} does not start with its count leaf "
                    f"name ({leaf_count.member('fName')!r})"
                )
            counter_dim = groups[0]
            numeric_groups = groups[1:]
        for group in numeric_groups:
            if not group.isdigit():
                raise ValueError(
                    f"the title of {where} has a non-numeric dimension "
                    f"[{group}] after the counter dimension"
                )
        shape = tuple(int(group) for group in numeric_groups)

        fLen = leaf.member("fLen")
        expected_fLen = 1
        for item in shape:
            expected_fLen *= item
        if fLen != expected_fLen:
            raise ValueError(
                f"the leaf of {where} has fLen {fLen} but its title implies "
                f"{expected_fLen} items per entry"
            )
        if leaf.member("fLenType") != dtype.itemsize:
            raise ValueError(
                f"the leaf of {where} has fLenType {leaf.member('fLenType')} "
                f"but its type {dtype!r} has itemsize {dtype.itemsize}"
            )
        if leaf.member("fOffset") != 0:
            raise unsupported(f"the leaf of {where} has a non-zero fOffset")

        # The leaf title must be the one the cascading writer regenerates.
        dims_text = "".join(f"[{group}]" for group in groups)
        if leaf.member("fTitle") != leaf.member("fName") + dims_text:
            raise unsupported(
                f"the leaf title {leaf.member('fTitle')!r} of {where} is not "
                "the generated name-plus-dimensions form"
            )

        if is_counter:
            if dtype != numpy.dtype(">i4"):
                raise unsupported(
                    f"counter branch {bname!r} of TTree {old_name!r} in file "
                    f"{file_path} is not a signed 32-bit integer"
                )
        elif leaf.member("fIsRange"):
            raise unsupported(
                f"branch {bname!r} of TTree {old_name!r} in file {file_path} "
                "has fIsRange set but is not a counter branch"
            )

        # Basket locator table checks.
        if len(branch.embedded_baskets) != 0:
            raise unsupported(f"branch {bname!r} has embedded TBaskets")
        branch_filename = branch.member("fFileName", none_if_missing=True)
        if branch_filename:
            raise unsupported(
                f"branch {bname!r} stores its baskets in another file "
                f"({branch_filename!r})"
            )
        branch_num_baskets = branch._num_normal_baskets
        if branch.member("fWriteBasket") != branch_num_baskets:
            raise ValueError(
                f"fWriteBasket is {branch.member('fWriteBasket')} but "
                f"{branch_num_baskets} baskets were found in {where}"
            )
        if num_baskets is None:
            num_baskets = branch_num_baskets
        elif branch_num_baskets != num_baskets:
            raise unsupported(
                f"branch {bname!r} has {branch_num_baskets} baskets but other "
                f"branches have {num_baskets} (branches with different basket "
                "counts cannot be copied)"
            )
        if branch.member("fEntries") != tree_entries:
            raise unsupported(
                f"branch {bname!r} has {branch.member('fEntries')} entries "
                f"but the TTree has {tree_entries} (branches that are not in "
                "sync cannot be copied)"
            )

        fBasketEntry = branch.member("fBasketEntry")
        if len(fBasketEntry) < branch_num_baskets + 1:
            raise ValueError(
                f"fBasketEntry has {len(fBasketEntry)} slots for "
                f"{branch_num_baskets} baskets in {where}"
            )
        basket_entry = [int(x) for x in fBasketEntry[: branch_num_baskets + 1]]
        if basket_entry[0] != 0:
            raise ValueError(f"fBasketEntry does not start at 0 in {where}")
        if any(
            basket_entry[i] > basket_entry[i + 1]
            for i in range(branch_num_baskets)
        ):
            raise ValueError(f"fBasketEntry is not monotonic in {where}")
        if basket_entry[branch_num_baskets] != tree_entries:
            raise ValueError(
                f"fBasketEntry ends at {basket_entry[branch_num_baskets]} but "
                f"the branch has {tree_entries} entries in {where}"
            )

        fBasketSeek = branch.member("fBasketSeek")
        fBasketBytes = branch.member("fBasketBytes")
        if len(fBasketSeek) < branch_num_baskets or len(fBasketBytes) < branch_num_baskets:
            raise ValueError(f"basket locator arrays are too short in {where}")

        plan = _BranchCopyPlan(bname)
        plan.is_counter = is_counter
        plan.fCompress = int(branch.member("fCompress"))
        plan.fBasketSize = int(branch.member("fBasketSize"))
        plan.fEntryOffsetLen = int(branch.member("fEntryOffsetLen"))
        plan.fOffset = int(branch.member("fOffset"))
        plan.fSplitLevel = int(branch.member("fSplitLevel"))
        plan.fFirstEntry = int(_member(branch, "fFirstEntry", 0))
        plan.fTotBytes = int(branch.member("fTotBytes"))
        plan.fZipBytes = int(branch.member("fZipBytes"))
        plan.tleaf_minimum = _member(leaf, "fMinimum", 0)
        plan.tleaf_maximum = _member(leaf, "fMaximum", 0)
        plan.basket_entry = basket_entry
        plan.basket_seek = []
        plan.basket_bytes = []
        plan.basket_big = []

        branch_tot_bytes = 0
        branch_zip_bytes = 0
        name_bytes = bname.encode(errors="surrogateescape")
        head_window = 64 + len(name_bytes)

        for i in range(branch_num_baskets):
            seek = int(fBasketSeek[i])
            nbytes = int(fBasketBytes[i])
            if seek <= 0 or nbytes <= 0:
                raise ValueError(
                    f"basket {i} of {where} has non-positive position or "
                    f"size (fBasketSeek={seek}, fBasketBytes={nbytes})"
                )
            if seek + nbytes > file.fEND:
                raise ValueError(
                    f"basket {i} of {where} refers to bytes "
                    f"[{seek}, {seek + nbytes}) beyond the end of the source "
                    f"file ({file.fEND})"
                )
            chunk = file.source.chunk(seek, seek + min(nbytes, head_window))
            head = _parse_basket_head(
                uproot._util.tobytes(chunk.raw_data),
                seek,
                nbytes,
                bname,
                old_name,
                file_path,
            )
            branch_tot_bytes += head.fKeylen + head.fObjlen
            branch_zip_bytes += nbytes
            if not head.big:
                has_small_keys = True
            plan.basket_seek.append(seek)
            plan.basket_bytes.append(nbytes)
            plan.basket_big.append(head.big)

        if branch_tot_bytes != plan.fTotBytes:
            raise ValueError(
                f"the baskets of {where} add up to {branch_tot_bytes} "
                f"uncompressed bytes but fTotBytes is {plan.fTotBytes}"
            )
        if branch_zip_bytes != plan.fZipBytes:
            raise ValueError(
                f"the baskets of {where} add up to {branch_zip_bytes} "
                f"compressed bytes but fZipBytes is {plan.fZipBytes}"
            )

        sum_tot_bytes += plan.fTotBytes
        sum_zip_bytes += plan.fZipBytes
        total_payload_bytes += branch_zip_bytes
        max_basket_capacity = max(
            max_basket_capacity, int(branch.member("fMaxBaskets"))
        )
        plans.append(plan)

        # Type specification for rebuilding this branch (counters are
        # re-created automatically by the cascading writer).
        if not is_counter:
            if counter_dim is None:
                if shape == ():
                    branch_types[bname] = dtype
                else:
                    branch_types[bname] = numpy.dtype((dtype, shape))
            else:
                counter_names[bname] = counter_dim
                inner = "".join(f"{item} * " for item in shape)
                branch_types[bname] = f"var * {inner}{dtype.name}"

    if num_baskets is None:
        num_baskets = 0

    tree_tot_bytes = int(_member(tree, "fTotBytes", 0))
    tree_zip_bytes = int(_member(tree, "fZipBytes", 0))
    if sum_tot_bytes != tree_tot_bytes:
        raise ValueError(
            f"the branches of TTree {old_name!r} in file {file_path} add up "
            f"to {sum_tot_bytes} uncompressed bytes but the TTree fTotBytes "
            f"is {tree_tot_bytes}"
        )
    if sum_zip_bytes != tree_zip_bytes:
        raise ValueError(
            f"the branches of TTree {old_name!r} in file {file_path} add up "
            f"to {sum_zip_bytes} compressed bytes but the TTree fZipBytes is "
            f"{tree_zip_bytes}"
        )

    plan = _TreeCopyPlan()
    plan.source = file.source
    plan.file_path = file_path
    plan.tree_title = tree.title
    plan.metadata = {
        "fTotBytes": tree_tot_bytes,
        "fZipBytes": tree_zip_bytes,
        "fSavedBytes": int(_member(tree, "fSavedBytes", 0)),
        "fFlushedBytes": int(_member(tree, "fFlushedBytes", 0)),
        "fWeight": float(_member(tree, "fWeight", 1.0)),
        "fTimerInterval": int(_member(tree, "fTimerInterval", 0)),
        "fScanField": int(_member(tree, "fScanField", 25)),
        "fUpdate": int(_member(tree, "fUpdate", 0)),
        "fDefaultEntryOffsetLen": int(_member(tree, "fDefaultEntryOffsetLen", 1000)),
        "fNClusterRange": 0,
        "fMaxEntries": int(_member(tree, "fMaxEntries", 1000000000000)),
        "fMaxEntryLoop": int(_member(tree, "fMaxEntryLoop", 1000000000000)),
        "fMaxVirtualSize": int(_member(tree, "fMaxVirtualSize", 0)),
        "fAutoSave": int(_member(tree, "fAutoSave", -300000000)),
        "fAutoFlush": int(_member(tree, "fAutoFlush", -30000000)),
        "fEstimate": int(_member(tree, "fEstimate", 1000000)),
    }
    plan.num_entries = int(tree_entries)
    plan.num_baskets = num_baskets
    plan.capacity = max(max_basket_capacity, num_baskets + 2)
    plan.branch_types = branch_types
    plan.counter_names = counter_names
    plan.branches = plans
    plan.total_payload_bytes = total_payload_bytes
    plan.metadata_bytes_estimate = 4096 + len(plans) * (512 + plan.capacity * 20)
    plan.has_small_keys = has_small_keys
    return plan


def check_fits_small_keys(dest_file, tree_plans, other_bytes=0):
    """
    Conservative guard: baskets whose source TKey uses 4-byte (small) seek
    pointers can only be copied to destination positions below
    ``uproot.const.kStartBigFile``, because rewriting the key with 8-byte
    pointers would change ``fKeylen`` and invalidate offsets stored inside the
    (verbatim-copied) compressed payload. Raises ``NotImplementedError`` if
    the destination file might grow past that boundary during this copy.
    """
    if not any(plan.has_small_keys for _, plan in tree_plans):
        return
    estimate = other_bytes + sum(
        plan.total_payload_bytes + plan.metadata_bytes_estimate
        for _, plan in tree_plans
    )
    end = dest_file._cascading.fileheader.end
    if end + estimate >= uproot.const.kStartBigFile:
        raise NotImplementedError(
            "copy_from cannot copy TBaskets with 4-byte TKey pointers into a "
            "file that may grow beyond 2 GB: the destination basket positions "
            "would not fit in the source key format (estimated new bytes: "
            f"{estimate}, current file end: {end})"
        )


def _copy_basket_payload(source, sink, seek, location, num_bytes, big, parent_location):
    """
    Copies one TBasket's bytes from ``source`` at ``seek`` to ``sink`` at
    ``location``, patching only the TKey's fSeekKey/fSeekPdir fields. Every
    read and write request is at most MAX_COPY_REQUEST bytes.
    """
    first = True
    src = seek
    dst = location
    remaining = num_bytes
    while remaining > 0:
        num = min(MAX_COPY_REQUEST, remaining)
        chunk = source.chunk(src, src + num)
        data = chunk.raw_data
        if first:
            # the first slice always covers the whole TKey header, because
            # num_bytes >= fKeylen > 34
            head = bytearray(data)
            if big:
                head[18:26] = struct.pack(">q", dst)
                head[26:34] = struct.pack(">q", parent_location)
            else:
                head[18:22] = struct.pack(">i", dst)
                head[22:26] = struct.pack(">i", parent_location)
            sink.write(dst, head)
            first = False
        else:
            sink.write(dst, data)
        src += num
        dst += num
        remaining -= num


def execute_tree_copy(directory, new_name, plan):
    """
    Copies one planned TTree into ``directory`` (a
    :doc:`uproot.writing.writable.WritableDirectory`) under ``new_name`` and
    returns the :doc:`uproot.writing._cascadetree.Tree` (already written to
    the file). All baskets are copied first; the TTree metadata and its
    directory key are written last, so an interrupted copy never leaves a
    readable but incomplete TTree behind.
    """
    cascading = directory._cascading
    sink = directory._file.sink
    freesegments = cascading.freesegments

    tree = uproot.writing._cascadetree.Tree(
        cascading,
        new_name,
        plan.tree_title,
        plan.branch_types,
        freesegments,
        lambda counted: plan.counter_names[counted],
        lambda outer, inner: inner if outer == "" else outer + "_" + inner,
        plan.capacity,
        10.0,
    )

    for branch_plan in plan.branches:
        datum = tree._branch_data[tree._branch_lookup[branch_plan.name]]
        datum["fCompress"] = branch_plan.fCompress
        if branch_plan.fCompress != 0:
            datum["compression"] = uproot.compression.Compression.from_code(
                branch_plan.fCompress
            )
        datum["fBasketSize"] = branch_plan.fBasketSize
        datum["fEntryOffsetLen"] = branch_plan.fEntryOffsetLen
        datum["fOffset"] = branch_plan.fOffset
        datum["fSplitLevel"] = branch_plan.fSplitLevel
        datum["fFirstEntry"] = branch_plan.fFirstEntry
        datum["fTotBytes"] = branch_plan.fTotBytes
        datum["fZipBytes"] = branch_plan.fZipBytes
        datum["tleaf_minimum_value"] = branch_plan.tleaf_minimum
        datum["tleaf_maximum_value"] = branch_plan.tleaf_maximum
        datum["fBasketEntry"][: plan.num_baskets + 1] = branch_plan.basket_entry
        datum["fBasketBytes"][: plan.num_baskets] = branch_plan.basket_bytes

    tree._metadata.update(plan.metadata)
    tree._num_entries = plan.num_entries
    tree._num_baskets = plan.num_baskets

    parent_location = cascading.key.location
    allocated = []
    try:
        for branch_plan in plan.branches:
            datum = tree._branch_data[tree._branch_lookup[branch_plan.name]]
            for i in range(plan.num_baskets):
                num_bytes = branch_plan.basket_bytes[i]
                location = freesegments.allocate(num_bytes, dry_run=False)
                allocated.append((location, location + num_bytes))
                _copy_basket_payload(
                    plan.source,
                    sink,
                    branch_plan.basket_seek[i],
                    location,
                    num_bytes,
                    branch_plan.basket_big[i],
                    parent_location,
                )
                datum["fBasketSeek"][i] = location

        tree.write_anew(sink)

    except BaseException:
        # Best-effort cleanup: return this tree's not-yet-referenced
        # allocations to the in-memory free-segment list. The destination
        # file on disk was never given a key for this tree. Cleanup errors
        # must not mask the original exception.
        for start, stop in allocated:
            with contextlib.suppress(Exception):
                freesegments.release(start, stop)
        raise

    return tree
