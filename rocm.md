# rocm.md

**Status of this file: guidance for the DEFERRED HIP lane.**
Nothing in this file has been exercised by this repository's QA. The five
`src/**/*_hip.hip` translation units have never been compiled on this host, so there
is no HIP build or HIP test result to cite — see
[`docs/CUDA_HIP_DEFERRED_CHANGES.md`](docs/CUDA_HIP_DEFERRED_CHANGES.md)
(audits `cross-backend:C1`, `cross-backend:C3`) for the compile gate a future HIP lane
must establish first, and `README.md#backend-status` for what is actually built here
(CPU/OpenMP only, `ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure`).
These porting rules are the input to that future lane, not evidence about the present.

**⚠️ CUDA/NVIDIA Path Status: DEFERRED**
The CUDA (NVIDIA) code path is **untested and may not compile**; there is no NVIDIA
hardware here, and the code is not actively maintained. The CPU backend is the one
that is built and tested. HIP is *not* a working alternative to fall back on — it is
deferred for the same reason (never compiled), which is what the earlier "use HIP
instead" wording in these docs got wrong.

## Goal

Write correct, maintainable HIP/ROCm code.

Priority:

```text
Correctness
> Memory locality
> Throughput
> Occupancy
> Micro-optimization
```

---

## Porting Rules

* Preserve semantics.
* Preserve indexing.
* Preserve synchronization.
* Preserve memory layout until profiling justifies change.
* Port first. Optimize later.
* One change class at time.

Forbidden:

* Port + refactor.
* Port + optimization.
* Port + algorithm rewrite.

---

## Execution Model

Think wavefronts, not threads.

Questions:

* Coalesced access?
* Divergence?
* Register pressure?
* LDS reuse?
* Occupancy sufficient to hide latency?

---

## Memory Rules

Assume memory-bound until profiling proves otherwise.

Priority:

```text
Registers
> LDS
> Cache
> Global memory
```

Rules:

* Load once.
* Reuse many times.
* Write once.
* Keep neighboring lanes on neighboring addresses.
* Prefer contiguous reads/writes.
* Minimize atomics.
* Minimize memory traffic before optimizing math.

---

## Divergence Rules

Avoid divergent hot paths.

Bad:

```cpp
if(condition_per_lane)
```

Prefer:

* uniform branches
* predication
* work partitioning
* separate kernels

---

## Occupancy Rules

Occupancy ≠ performance.

Watch:

* VGPR usage
* LDS usage
* spills
* wave residency

Higher occupancy only matters if latency hidden better.

---

## Synchronization Rules

Use smallest correct scope.

* No barrier without reason.
* No atomics without reason.
* No host sync without dependency.

---

## Agent Rules

Do not:

* invent optimizations
* change numerical behavior
* remove barriers blindly
* change launch geometry without reason
* replace clear code with clever code

Do:

* explain non-trivial changes
* keep generated code compilable
* preserve behavior
* profile-driven optimize

---

## Profiling Rules

Never claim performance improvement without measurement.

Check:

* kernel time
* memory throughput
* occupancy
* VGPR pressure
* LDS pressure
* divergence
* atomic contention

---

## Cardinal Sins

* Random memory access
* Divergent hot paths
* Excessive atomics
* Register spills
* Tiny kernels
* Synchronization abuse
* Premature optimization
* CPU-style thinking

---

## Default Assumption

Problem memory-bound until profiler says otherwise.
