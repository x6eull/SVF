//===- Options.cpp -- Command line options ------------------------//

#include "Util/Options.h"
#include "Util/CommandLine.h"
#include "FastCluster/fastcluster.h"
#include "Util/ExtAPI.h"
#include "MSSA/MemSSA.h"
#include "WPA/WPAPass.h"
#include "AE/Svfexe/AbstractInterpretation.h"

namespace SVF
{
const OptionMap<enum PTAStat::ClockType> Options::ClockType(
    "clock-type",
    "how time should be measured",
    PTAStat::ClockType::CPU,
{
    {PTAStat::ClockType::Wall, "wall", "use wall time"},
    {PTAStat::ClockType::CPU, "cpu", "use CPU time"},
}
);

const Option<bool> Options::MarkedClocksOnly(
    "marked-clocks-only",
    "Only measure times where explicitly marked",
    true
);

const OptionMap<NodeIDAllocator::Strategy> Options::NodeAllocStrat(
    "node-alloc-strat",
    "Method of allocating (LLVM) values and memory objects as node IDs",
    NodeIDAllocator::Strategy::SEQ,
{
    {NodeIDAllocator::Strategy::DENSE, "dense", "allocate objects together [0-n] and values together [m-MAX], separately"},
    {NodeIDAllocator::Strategy::REVERSE_DENSE, "reverse-dense", "like dense but flipped, objects are [m-MAX], values are [0-n]"},
    {NodeIDAllocator::Strategy::SEQ, "seq", "allocate values and objects sequentially, intermixed (default)"},
    {NodeIDAllocator::Strategy::DBUG, "debug", "allocate value and objects sequentially, intermixed, except GEP objects as offsets"},
}
);

const Option<u32_t> Options::MaxFieldLimit(
    "field-limit",
    "Maximum number of fields for field sensitive analysis",
    512
);

const OptionMap<BVDataPTAImpl::PTBackingType> Options::ptDataBacking(
    "ptd",
    "Overarching points-to data structure",
    BVDataPTAImpl::PTBackingType::Persistent,
{
    {BVDataPTAImpl::PTBackingType::Mutable, "mutable", "points-to set per pointer"},
    {BVDataPTAImpl::PTBackingType::Persistent, "persistent", "points-to set ID per pointer, operations hash-consed"},
}
);

const Option<u32_t> Options::FsTimeLimit(
    "fs-time-limit",
    "time limit for main phase of flow-sensitive analyses",
    0
);

const Option<u32_t> Options::VersioningThreads(
    "versioning-threads",
    "number of threads to use in the versioning phase of versioned flow-sensitive analysis",
    1
);

const Option<u32_t> Options::AnderTimeLimit(
    "ander-time-limit",
    "time limit for Andersen's analyses (ignored when -fs-time-limit set)",
    0
);

// ContextDDA.cpp
const Option<u32_t> Options::CxtBudget(
    "cxt-bg",
    "Maximum step budget of context-sensitive traversing",
    10000
);

// DDAPass.cpp
const Option<u32_t> Options::MaxPathLen(
    "max-path",
    "Maximum path limit for DDA",
    100000
);

const Option<u32_t> Options::MaxContextLen(
    "max-cxt",
    "Maximum context limit for DDA",
    3
);

const Option<u32_t> Options::MaxStepInWrapper(
    "max-step",
    "Maximum steps when traversing on SVFG to identify a memory allocation wrapper",
    10
);

const Option<std::string> Options::UserInputQuery(
    "query",
    "Please specify queries by inputing their pointer ids",
    "all"
);

const Option<bool> Options::InsenRecur(
    "in-recur",
    "Mark context insensitive SVFG edges due to function recursions",
    false
);

const Option<bool> Options::InsenCycle(
    "in-cycle",
    "Mark context insensitive SVFG edges due to value-flow cycles",
    false
);

const Option<bool> Options::PrintCPts(
    "cpts",
    "Dump conditional points-to set ",
    false
);

const Option<bool> Options::PrintQueryPts(
    "print-query-pts",
    "Dump queries' conditional points-to set ",
    false
);

const Option<bool> Options::WPANum(
    "wpa-num",
    "collect WPA FS number only ",
    false
);

/// register this into alias analysis group
//static RegisterAnalysisGroup<AliasAnalysis> AA_GROUP(DDAPA);
OptionMultiple<PointerAnalysis::PTATY> Options::DDASelected(
    "Select pointer analysis",
{
    {PointerAnalysis::FlowS_DDA, "dfs", "Demand-driven flow sensitive analysis"},
    {PointerAnalysis::Cxt_DDA, "cxt", "Demand-driven context- flow- sensitive analysis"},
}
);

// FlowDDA.cpp
const Option<u32_t> Options::FlowBudget(
    "flow-bg",
    "Maximum step budget of flow-sensitive traversing",
    10000
);


// Offline constraint graph (OfflineConsG.cpp)
const Option<bool> Options::OCGDotGraph(
    "dump-ocg",
    "Dump dot graph of Offline Constraint Graph",
    false
);


// Program Assignment Graph for pointer analysis (SVFIR.cpp)
Option<bool> Options::HandBlackHole(
    "blk",
    "Handle blackhole edge",
    false
);

const Option<bool> Options::FirstFieldEqBase(
    "ff-eq-base",
    "Treat base objects as their first fields",
    false
);


// SVFG optimizer (SVFGOPT.cpp)
const Option<bool> Options::ContextInsensitive(
    "ci-svfg",
    "Reduce SVFG into a context-insensitive one",
    false
);

const Option<bool> Options::KeepAOFI(
    "keep-aofi",
    "Keep formal-in and actual-out parameters",
    false
);

const Option<std::string> Options::SelfCycle(
    "keep-self-cycle",
    "How to handle self cycle edges: all, context, none",
    ""
);


// Sparse value-flow graph (VFG.cpp)
const Option<bool> Options::DumpVFG(
    "dump-vfg",
    "Dump dot graph of VFG",
    false
);


// Base class of pointer analyses (PointerAnalysis.cpp)
const Option<bool> Options::TypePrint(
    "print-type",
    "Print type",
    false
);

const Option<bool> Options::FuncPointerPrint(
    "print-fp",
    "Print targets of indirect call site",
    false
);

const Option<bool> Options::PTSPrint(
    "print-pts",
    "Print points-to set of top-level pointers",
    false
);

const Option<bool> Options::PrintFieldWithBasePrefix(
    "print-field",
    "Print field object with base object id as the prefix",
    false
);

const Option<bool> Options::PTSAllPrint(
    "print-all-pts",
    "Print all points-to set of both top-level and address-taken variables",
    false
);

const Option<bool> Options::PStat(
    "stat",
    "Statistic for Pointer analysis",
    true
);

const Option<u32_t> Options::StatBudget(
    "stat-limit",
    "Iteration budget for On-the-fly statistics",
    20
);

const Option<bool> Options::PAGDotGraph(
    "dump-pag",
    "Dump dot graph of SVFIR",
    false
);

const Option<bool> Options::ShowSVFIRValue(
    "show-ir-value",
    "Show values of SVFIR (e.g., when generating dot graph)",
    true
);

const Option<bool> Options::DumpICFG(
    "dump-icfg",
    "Dump dot graph of ICFG",
    false
);

const Option<std::string> Options::DumpJson(
    "dump-json",
    "Dump the SVFIR in JSON format",
    ""
);

const Option<bool> Options::ReadJson(
    "read-json",
    "Read the SVFIR in JSON format",
    false
);

const Option<bool> Options::CallGraphDotGraph(
    "dump-callgraph",
    "Dump dot graph of Call Graph",
    false
);

const Option<bool> Options::PAGPrint(
    "print-pag",
    "Print SVFIR to command line",
    false
);

const Option<u32_t> Options::IndirectCallLimit(
    "ind-call-limit",
    "Indirect solved call edge limit",
    50000
);

Option<bool> Options::UsePreCompFieldSensitive(
    "pre-field-sensitive",
    "Use pre-computed field-sensitivity for later analysis",
    true
);

const Option<bool> Options::EnableAliasCheck(
    "alias-check",
    "Enable alias check functions",
    true
);

const Option<bool> Options::EnableTypeCheck(
    "type-check",
    "Enable type check functions",
    true
);

const Option<bool> Options::EnableThreadCallGraph(
    "enable-tcg",
    "Enable pointer analysis to use thread call graph",
    true
);

const Option<bool> Options::ConnectVCallOnCHA(
    "v-call-cha",
    "connect virtual calls using cha",
    false
);


// PointerAnalysisImpl.cpp
const Option<bool> Options::INCDFPTData(
    "inc-data",
    "Enable incremental DFPTData for flow-sensitive analysis",
    true
);

const Option<bool> Options::ClusterAnder(
    "cluster-ander",
    "Stage Andersen's with Steensgard's and cluster based on that",
    false
);

const Option<bool> Options::ClusterFs(
    "cluster-fs",
    "Cluster for FS/VFS with auxiliary Andersen's",
    false
);

const Option<bool> Options::PlainMappingFs(
    "plain-mapping-fs",
    "Use an explicitly (not null) plain mapping for FS",
    false
);

const OptionMap<PointsTo::Type> Options::PtType(
    "pt-type",
    "points-to set data structure to use in all analyses",
    PointsTo::Type::SBV,
{
    {PointsTo::Type::SBV, "sbv", "sparse bit-vector"},
    {PointsTo::Type::CBV, "cbv", "core bit-vector (dynamic bit-vector without leading and trailing 0s)"},
    {PointsTo::Type::BV, "bv", "bit-vector (dynamic bit-vector without trailing 0s)"},
    {PointsTo::Type::SEGBV, "segbv", "segment bit-vector (dynamic bit-vector with segments)"},
}
);

const OptionMap<u32_t> Options::ClusterMethod(
    "cluster-method",
    "hierarchical clustering method for objects",
    HCLUST_METHOD_SVF_BEST,
{
    {HCLUST_METHOD_SINGLE,     "single", "single linkage; minimum spanning tree algorithm"},
    {HCLUST_METHOD_COMPLETE, "complete", "complete linkage; nearest-neighbour-chain algorithm"},
    {HCLUST_METHOD_AVERAGE,   "average", "unweighted average linkage; nearest-neighbour-chain algorithm"},
    {HCLUST_METHOD_SVF_BEST,     "best", "try all linkage criteria; choose best"},
}
);

const Option<bool> Options::RegionedClustering(
    // Use cluster to "gather" the options closer together, even if it sounds a little worse.
    "cluster-regioned",
    "cluster regions separately",
    true
);

const Option<bool> Options::RegionAlign(
    "cluster-region-aligned",
    "align each region's identifiers to the native word size",
    true
);

const Option<bool> Options::PredictPtOcc(
    "cluster-predict-occ",
    "try to predict which points-to sets are more important in staged analysis",
    false
);

// Memory region (MemRegion.cpp)
const Option<bool> Options::IgnoreDeadFun(
    "mssa-ignore-dead-fun",
    "Don't construct memory SSA for deadfunction",
    false
);


// Base class of pointer analyses (MemSSA.cpp)
const Option<bool> Options::DumpMSSA(
    "dump-mssa",
    "Dump memory SSA",
    false
);

const Option<std::string> Options::MSSAFun(
    "mssa-fun",
    "Please specify which function needs to be dumped",
    ""
);

const OptionMap<u32_t> Options::MemPar(
    "mem-par",
    "Memory region partition strategies (e.g., for SVFG construction)",
    MemSSA::MemPartition::IntraDisjoint,
{
    {MemSSA::MemPartition::Distinct, "distinct", "memory region per each object"},
    {MemSSA::MemPartition::IntraDisjoint, "intra-disjoint", "memory regions partitioned based on each function"},
    {MemSSA::MemPartition::InterDisjoint, "inter-disjoint", "memory regions partitioned across functions"},
}
);


// SVFG builder (SVFGBuilder.cpp)
const Option<bool> Options::SVFGWithIndirectCall(
    "svfg-with-ind-call",
    "Update Indirect Calls for SVFG using pre-analysis",
    false
);

Option<bool> Options::OPTSVFG(
    "opt-svfg",
    "Optimize SVFG to eliminate formal-in and actual-out",
    false
);

const Option<std::string> Options::WriteSVFG(
    "write-svfg",
    "Write SVFG's analysis results to a file",
    ""
);

const Option<std::string> Options::ReadSVFG(
    "read-svfg",
    "Read SVFG's analysis results from a file",
    ""
);


const Option<bool> Options::IntraLock(
    "intra-lock-td-edge",
    "Use simple intra-procedural lock for adding SVFG edges",
    true
);


// LockAnalysis.cpp
const Option<bool> Options::PrintLockSpan(
    "print-lock",
    "Print Thread Interleaving Results",
    false
);


// MHP.cpp
const Option<bool> Options::PrintInterLev(
    "print-interlev",
    "Print Thread Interleaving Results",
    false
);

const Option<bool> Options::DoLockAnalysis(
    "lock-analysis",
    "Run Lock Analysis",
    true
);


// MTAStat.cpp
const Option<bool> Options::AllPairMHP(
    "all-pair-mhp",
    "All pair MHP computation",
    false
);


// TCT.cpp
const Option<bool> Options::TCTDotGraph(
    "dump-tct",
    "Dump dot graph of Call Graph",
    false
);


// LeakChecker.cpp
const Option<bool> Options::ValidateTests(
    "valid-tests",
    "Validate memory leak tests",
    false
);


// Source-sink analyzer (SrcSnkDDA.cpp)
const Option<bool> Options::DumpSlice(
    "dump-slice",
    "Dump dot graph of Saber Slices",
    false
);

const Option<u32_t> Options::CxtLimit(
    "cxt-limit",
    "Source-Sink Analysis Contexts Limit",
    3
);


// CHG.cpp
const Option<bool> Options::DumpCHA(
    "dump-cha",
    "dump the class hierarchy graph",
    false
);


// DCHG.cpp
const Option<bool> Options::PrintDCHG(
    "print-dchg",
    "print the DCHG if debug information is available",
    false
);


// LLVMModule.cpp
const Option<std::string> Options::Graphtxt(
    "graph-txt",
    "graph txt file to build SVFIR",
    ""
);

const Option<bool> Options::SVFMain(
    "svf-main",
    "add svf.main()",
    false
);

Option<bool> Options::ModelConsts(
    "model-consts",
    "Modeling individual constant objects",
    false
);

Option<bool> Options::ModelArrays(
    "model-arrays",
    "Modeling Gep offsets for array accesses",
    false
);

const Option<bool> Options::CyclicFldIdx(
    "cyclic-field-index",
    "Enable cyclic field index when generating field objects using modulus offset",
    false
);

const Option<bool> Options::SymTabPrint(
    "print-symbol-table",
    "Print Symbol Table to command line",
    false
);

// Conditions.cpp
const Option<u32_t> Options::MaxZ3Size(
    "max-z3-size",
    "Maximum size limit for Z3 expression",
    30
);

// BoundedZ3Expr.cpp
const Option<u32_t> Options::MaxBVLen(
    "max-bv-len",
    "Maximum length limit for Z3 bitvector",
    64
);



// SaberCondAllocator.cpp
const Option<bool> Options::PrintPathCond(
    "print-pc",
    "Print out path condition",
    false
);


// SaberSVFGBuilder.cpp
const Option<bool> Options::CollectExtRetGlobals(
    "saber-collect-extret-globals",
    "Don't include pointers returned by external function during collecting globals",
    true
);


// SVFUtil.cpp
const Option<bool> Options::DisableWarn(
    "dwarn",
    "Disable warning",
    true
);


// Andersen.cpp
const Option<bool> Options::ConsCGDotGraph(
    "dump-constraint-graph",
    "Dump dot graph of Constraint Graph",
    false
);

const Option<bool> Options::BriefConsCGDotGraph(
    "brief-constraint-graph",
    "Dump dot graph of Constraint Graph",
    true
);

const Option<bool> Options::PrintCGGraph(
    "print-constraint-graph",
    "Print Constraint Graph to Terminal",
    false
);

const Option<std::string> Options::WriteAnder(
    "write-ander",
    "-write-ander=ir_annotator (Annotated IR with Andersen's results) or write Andersen's analysis results to a user-specified text file",
    ""
);

const Option<std::string> Options::ReadAnder(
    "read-ander",
    "-read-ander=ir_annotator (Read Andersen's analysis results from the annotated IR, e.g., *.pre.bc) or from a text file",
    ""
);

const Option<bool> Options::DiffPts(
    "diff",
    "Enable differential point-to set",
    true
);

Option<bool> Options::DetectPWC(
    "merge-pwc",
    "Enable PWC detection",
    true
);

//SVFIRBuilder.cpp
const Option<bool> Options::VtableInSVFIR(
    "vt-in-ir",
    "Handle vtable in ConstantArray/ConstantStruct in SVFIRBuilder (already handled in CHA?)",
    false
);

//WPAPass.cpp
const Option<std::string> Options::ExtAPIPath(
    "extapi",
    "External API extapi.bc",
    ""
);

const Option<bool> Options::AnderSVFG(
    "svfg",
    "Generate SVFG after Andersen's Analysis",
    false
);

const Option<bool> Options::SABERFULLSVFG(
    "saber-full-svfg",
    "When using SABER for bug detection pass, enable full svfg on top of the pointer-only one",
    false
);

const Option<bool> Options::PrintAliases(
    "print-aliases",
    "Print results for all pair aliases",
    false
);

OptionMultiple<PointerAnalysis::PTATY> Options::PASelected(
    "Select pointer analysis",
{
    {PointerAnalysis::Andersen_WPA, "nander", "Standard inclusion-based analysis"},
    {PointerAnalysis::AndersenSCD_WPA, "sander", "Selective cycle detection inclusion-based analysis"},
    {PointerAnalysis::AndersenSFR_WPA, "sfrander", "Stride-based field representation inclusion-based analysis"},
    {PointerAnalysis::AndersenWaveDiff_WPA, "ander", "Diff wave propagation inclusion-based analysis"},
    {PointerAnalysis::Steensgaard_WPA, "steens", "Steensgaard's pointer analysis"},
    // Disabled till further work is done.
    {PointerAnalysis::FSSPARSE_WPA, "fspta", "Sparse flow sensitive pointer analysis"},
    {PointerAnalysis::VFS_WPA, "vfspta", "Versioned sparse flow-sensitive points-to analysis"},
    {PointerAnalysis::TypeCPP_WPA, "type", "Type-based fast analysis for Callgraph, SVFIR and CHA"},
}
);


OptionMultiple<u32_t> Options::AliasRule(
    "Select alias check rule",
{
    {WPAPass::Conservative, "conservative", "return MayAlias if any pta says alias"},
    {WPAPass::Veto, "veto", "return NoAlias if any pta says no alias"},
}
);

const Option<bool> Options::ShowHiddenNode(
    "show-hidden-nodes",
    "Show hidden nodes on DOT Graphs (e.g., isolated node on a graph)",
    false
);

const Option<std::string> Options::GrammarFilename(
    "grammar",
    "<Grammar textfile>",
    ""
);

const Option<std::string> Options::CFLGraph(
    "cflgraph",
    "<Dot file as the CFLGraph input>",
    ""
);

const Option<bool> Options::PrintCFL(
    "print-cfl",
    "Print ir, grammar and cflgraph for debug.",
    false
);

const Option<bool> Options::FlexSymMap(
    "flex-symmap",
    "Extend exist sym map while read graph from dot if sym not in map.",
    false
);

const Option<bool> Options::PEGTransfer(
    "peg-transfer",
    "When explicit to true, cfl graph builder will transfer PAG load and store edges to copy and addr.",
    false
);

const Option<bool> Options::CFLSVFG(
    "cflsvfg",
    "When explicit to true, cfl graph builder will transfer SVFG to CFL Reachability.",
    false
);

const Option<bool> Options::POCRAlias(
    "pocr-alias",
    "When explicit to true, cfl data builder will transfer CFL graph to CFLData.",
    false
);

const Option<bool> Options::POCRHybrid(
    "pocr-hybrid",
    "When explicit to true, POCRHybridSolver transfer CFL graph to internal hybrid graph representation.",
    false
);

const Option<bool> Options::Customized(
    "customized",
    "When explicit to true, user can use any grammar file.",
    false
);

const Option<bool> Options::LoopAnalysis(
    "loop-analysis",
    "Analyze every func and get loop info and loop bounds.",
    true
);

const Option<u32_t> Options::LoopBound(
    "loop-bound",
    "Maximum number of loop",
    1
);

const Option<u32_t> Options::WidenDelay(
    "widen-delay", "Loop Widen Delay", 3);
const OptionMap<u32_t> Options::HandleRecur(
    "handle-recur",
    "Recursion handling mode in abstract execution (Default -widen-narrow)",
    AbstractInterpretation::HandleRecur::WIDEN_NARROW,
{
    {
        AbstractInterpretation::HandleRecur::TOP, "top",
        "The return value, and any stored object pointed by q at *q = p in recursive functions will be set to the top value."
    },
    {
        AbstractInterpretation::HandleRecur::WIDEN_ONLY, "widen-only",
        "Only apply widening at the cycle head of recursive functions."
    },
    {
        AbstractInterpretation::HandleRecur::WIDEN_NARROW, "widen-narrow",
        "Apply both widening and narrowing at the cycle head of recursive functions."
    }
}
);
const Option<u32_t> Options::Timeout(
    "timeout", "time out (seconds), set -1 (no timeout), default 14400s",14400);
const Option<std::string> Options::OutputName(
    "output","output db file","output.db");
const Option<bool> Options::BufferOverflowCheck(
    "overflow","Buffer Overflow Detection",false);
const Option<bool> Options::NullDerefCheck(
    "null-deref","Null Pointer Dereference Detection",false);
const Option<bool> Options::MemoryLeakCheck(
    "leak", "Memory Leak Detection",false);
const Option<bool> Options::FileCheck(
    "fileck", "File Open/Close Detection",false);
const Option<bool> Options::DFreeCheck(
    "dfree", "Double Free Detection",false);
const Option<bool> Options::RaceCheck(
    "race", "Data race Detection",false);
const Option<bool> Options::GepUnknownIdx(
    "gep-unknown-idx","Skip Gep Unknown Index",false);
const Option<bool> Options::RunUncallFuncs(
    "run-uncall-fun","Skip Gep Unknown Index",false);
const Option<bool> Options::ICFGMergeAdjacentNodes(
    "icfg-merge-adjnodes","ICFG Simplification - Merge Adjacent Nodes in the Same Basic Block.",false);


const Option<u32_t> Options::AEPrecision(
    "precision",
    "symbolic abstraction precision for float",
    0
);

} // namespace SVF.
