# C++ 与计算机科学课程完整笔记

> 教学背景：以 Perplexity AI 生产代码 `a2a_dispatch_send.cu` 为贯穿主线，
> 系统学习 C++ 程序设计与计算机科学核心概念。
>
> 课程共 12 节，覆盖内存管理、类型系统、并发、数据结构四大模块。

---

## 课程地图

```
模块一：类型系统与抽象
  L4  模板特化          ExpertIterator<Fixed<N>> vs 通用版
  L5  虚函数 vs 模板     vtable 开销，if constexpr
  L6  constexpr         编译期计算链的起点

模块二：并发与同步
  L7  原子操作与内存序   std::atomic，memory_order，happens-before
  L8  互斥锁与条件变量   std::mutex，condition_variable，死锁

模块三：资源管理
  L9  RAII 与智能指针   unique_ptr，shared_ptr，移动语义

模块四：数据结构与算法
  L10 迭代器模式        自定义迭代器，range-based for
  L11 前缀和三种实现     串行 / Blelloch / Warp Shuffle
  L12 内存布局与缓存     行主序，Coalesced，AoS vs SoA，对齐
```

---

## 第四课：模板特化

### 核心概念

同一个类名，多套实现，编译器根据模板参数**自动选版本**：

```cpp
// 通用版（Primary Template）
template<typename NumExpertsPerTokenTy>
class ExpertIterator { ... };

// 偏特化（Partial Specialization）：T = Fixed<N> 时使用
template <size_t N>
class ExpertIterator<Fixed<N>> { ... };
```

### 三种特化形式

```cpp
// 通用模板
template<typename T>
class Box { void describe() { printf("Generic\n"); } };

// 偏特化：T 是 Fixed<N> 的情况（还有剩余参数 N）
template<size_t N>
class Box<Fixed<N>> { void describe() { printf("Fixed, N=%zu\n", N); } };

// 完全特化：T 精确为 int（没有剩余参数）
template<>
class Box<int> { void describe() { printf("Int\n"); } };

// 优先级：完全特化 > 偏特化 > 通用模板
Box<double>   b1;  // Generic
Box<Fixed<8>> b2;  // Fixed, N=8
Box<int>      b3;  // Int
```

### 生产代码里的意义

| 版本 | `operator[]` | 全局内存读 |
|------|-------------|---------|
| 通用版 `NotFixed` | 每次调用读 indices/weights | NUM_STEPS × K 次 |
| `Fixed<N>` 特化 | 构造时一次性读完，存入寄存器数组 | 仅构造时 K 次 |

`#pragma unroll(N)` 只在 N 是编译期常量时有效 → 只有特化版能用。

### 作业 Q&A

**Q1：以下代码 `a.value()` 和 `b.value()` 输出什么？**
```cpp
template<typename T> class Container { int value() { return 0; } };
template<typename T> class Container<T*> { int value() { return 1; } };
template<>           class Container<int*> { int value() { return 2; } };

Container<double*> a;  // → 1（偏特化）
Container<int*>    b;  // → 2（完全特化，优先级更高）
```
输出：`1 2`

---

**Q2：`ExpertIterator<Fixed<2>>` 的成员数组占多少寄存器？**

`experts_[2]`（2）+ `weights_[2]`（2）+ `offsets_[2]`（2）+ `positions_[2]`（2）= **8 个寄存器**。
N 是 constexpr → 编译器把数组展开成独立寄存器变量，不存 local memory。

---

**Q3：通用版 vs 特化版，K=2，NUM_STEPS=2，多读多少次全局内存？**

通用版：`operator[]` 每次读 ~4 次全局内存，调用 NUM_STEPS × K = 4 次 → 共 **16 次**。
特化版：scatter 循环中 0 次（全从寄存器读），构造时读 8 次。
**差距：多读 8 次全局内存（在 scatter 循环中）。**

---

## 第五课：虚函数 vs 模板多态

### vtable 的底层结构

```
对象内存布局：
┌─────────┐
│  vptr   │ → vtable（只读，存在数据段）
├─────────┤       ┌──────────────────┐
│  成员   │       │ virtual_func 指针 │
└─────────┘       └──────────────────┘

调用 obj->virtual_func()：
1. 读 obj 的 vptr              // 1次内存读
2. 在 vtable 里找函数地址       // 1次内存读
3. 间接跳转执行                // 间接跳转
```

### 两种多态对比

| | 虚函数（运行时）| 模板（编译时）|
|--|--|--|
| 类型确定时机 | 运行时 | 编译时 |
| 调用开销 | 2次内存读 + 间接跳转 | 零（内联）|
| 能否 `#pragma unroll` | ❌ | ✅ |
| GPU kernel 能用吗 | ❌（vtable 在 host 内存）| ✅ |
| 适用场景 | 插件系统、运行时策略 | kernel 内部、固定类型 |

### `if constexpr`：编译期分支

```cpp
template<typename T>
void print_type() {
    if constexpr (std::is_same_v<T, int>) {
        printf("integer\n");     // 只有 T=int 时这行存在
    } else if constexpr (std::is_same_v<T, float>) {
        printf("float\n");
    } else {
        printf("unknown\n");
    }
}
// vs 普通 if：两个分支都会被编译器检查类型，可能报错
```

### 作业 Q&A

**Q1：`Animal a;` 能编译吗？**

不能。`Animal` 有纯虚函数 `= 0`，是抽象类，不能实例化。
`Animal *p = new Dog();` 合法（指针指向具体子类）。

---

**Q2：NUM_STEPS=2，K=2，改成虚函数后多出多少次额外内存读？**

scatter 循环调用 `operator[]` 共 4 次，每次虚函数调用 2 次额外读 → **8 次**额外内存读。
且 vtable 不在 L1 里，极大概率每次 cache miss，实际代价更高。

---

**Q3：用 `if constexpr` 实现 `print_type<T>()`**

```cpp
template<typename T>
void print_type() {
    if constexpr (std::is_same_v<T, int>)   printf("integer\n");
    else if constexpr (std::is_same_v<T, float>) printf("float\n");
    else                                     printf("unknown\n");
}
```

---

## 第六课：`constexpr` 与编译期计算

### `const` vs `constexpr`

```cpp
int n = get_from_user();
const int x = n * 2;          // 运行时常量，不可修改
constexpr int Y = 32;          // 编译期常量，值必须编译时确定

int arr[x];     // ❌ 非标准 VLA
int arr[Y];     // ✅ 编译期已知
```

| | `const` | `constexpr` |
|--|--|--|
| 值何时确定 | 运行时 | 编译时 |
| 用作数组大小 | ❌ | ✅ |
| 用作模板参数 | ❌ | ✅ |
| `#pragma unroll` 参数 | ❌ | ✅ |

### `constexpr` 函数

```cpp
constexpr size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

constexpr size_t NUM_STEPS = ceil_div(512, 32);  // 编译期求值 → 16
size_t x = get_input();
size_t result = ceil_div(x, 32);                 // 运行时求值，完全合法
```

### 完整优化链

```
constexpr TOKEN_DIM = 8192
    → Fixed<8192>（模板特化）
    → constexpr NUM_STEPS = ceil_div(...)
    → if constexpr (QUICK)（编译期分支）
    → uint4 vals[NUM_STEPS]（寄存器数组）
    → #pragma unroll(NUM_STEPS)（完全展开）
    → 零开销运行时路径
断开任何一环 → 下游所有优化全部失效
```

### `static_assert`：编译期断言

```cpp
static_assert(NUM_STEPS == 16, "expected 16");
// 不是编译期常量 → 报错；值不等于16 → 报错
```

### `#define` vs `constexpr`

```cpp
#define BAD 32 + 1
int arr[BAD * 2];  // 实际是 32 + 1*2 = 34，不是 66！文本替换陷阱

constexpr int GOOD = 32 + 1;
int arr[GOOD * 2]; // 66，类型安全，有作用域
```

### 作业 Q&A

**Q1：哪些行能编译？**
```cpp
const int a = 10; constexpr int b = 20;
int arr1[a];            // ❌ const 不是编译期常量
int arr2[b];            // ✅
constexpr int c = a+b;  // ❌ a 污染整个表达式
constexpr int d = a*2;  // ❌
static_assert(b == 20); // ✅
static_assert(a == 10); // ❌
```
规律：`constexpr` 有传染性，表达式里出现非 `constexpr` 变量，整个表达式就不是编译期常量。

---

**Q2：去掉 `constexpr` 后两行各报什么错？**

```cpp
size_t NUM_THREADS = ...; // 运行时变量

constexpr size_t NUM_STEPS = (...NUM_THREADS...);
// ❌ NUM_THREADS 不是常量表达式，无法初始化 constexpr 变量

uint4 vals[NUM_STEPS];
// ❌ 数组大小必须是编译期常量（VLA 在 C++ 中非标准）
```

---

**Q3：`static_assert` 检查 TOKEN_DIM 是 sizeof(uint4) 的倍数**

```cpp
static_assert(TOKEN_DIM % sizeof(uint4) == 0,
              "TOKEN_DIM must be a multiple of sizeof(uint4) for vectorized loads");
```

---

## 第七课：原子操作与内存序

### 竞态条件

```cpp
// counter++ 不是原子操作，实际是三步：读-改-写
// 两个线程同时执行，写可能丢失
```

### `std::atomic` 基本操作

```cpp
std::atomic<int> counter = 0;
counter.fetch_add(1);          // 原子加，返回旧值（= CUDA atomicAdd）
counter.load();                // 原子读
counter.store(42);             // 原子写
counter.compare_exchange_strong(expected, new_val);  // CAS
```

| C++ `std::atomic` | CUDA |
|--|--|
| `fetch_add(1)` | `atomicAdd(ptr, 1)` |
| `load()` | `ld_volatile_u32(ptr)` |
| `store(v)` | `st_mmio_b8(ptr, v)` |
| `compare_exchange_strong` | `atomicCAS` |

### 内存序（Memory Order）

```cpp
// 生产者（Release）
data = 42;
flag.store(1, std::memory_order_release);   // "之前的写，在你看到这个后都能看到"

// 消费者（Acquire）
while (flag.load(std::memory_order_acquire) == 0) {}
int val = data;  // 保证看到 42
```

| memory_order | 语义 | 适用 |
|--|--|--|
| `relaxed` | 只保证原子性 | 纯计数器 |
| `release` | store，之前的写可见 | 生产者 |
| `acquire` | load，之后的读有效 | 消费者 |
| `seq_cst` | 全局顺序（默认）| 最安全，最慢 |

### CUDA `grid.sync()` 的作用

= 全局 barrier（等待所有 block 到达）
\+ flush 所有 block 的 L1 写到 L2（release）
\+ 所有 block 重新从 L2 加载（acquire）

去掉它：Phase 2 的 block 读到 `expert_offsets` 的初始值 0，scatter 写到错误地址，静默数据损坏。

### 作业 Q&A

**Q1：`memory_order_relaxed` flag 的竞态**

有竞态。`relaxed` 允许编译器把 `flag.store` 重排到 `result = compute()` 之前，消费者看到 flag=1 时 result 还没写完。修复：生产者用 `release`，消费者用 `acquire`。

---

**Q2：`if (sum.load() < 100) { sum.fetch_add(val); }` 能保证 sum < 100 吗？**

不能。`load()` 和 `fetch_add()` 各自原子，但组合不原子。中间有间隙，多个线程都通过检查后都执行加法，sum 可能远超 100。
修复：用 CAS 循环，把"检查+修改"变成原子操作。

---

**Q3：去掉 `grid.sync()` 的后果**

三重问题叠加：
1. Phase 2 的 block 可能在 Phase 1 未完成时就开始执行
2. block 0 的写在 L1 里，其他 SM 上的 block 看不到（per-SM L1 不共享）
3. 结果是随机错误：不崩溃、不报错、只是 scatter 位置错误，每次运行错误模式不同——最难调试的 GPU bug。

---

## 第八课：互斥锁与条件变量

### `std::mutex` 与临界区

```cpp
std::mutex mtx;
int balance = 1000;

void transfer() {
    std::lock_guard<std::mutex> guard(mtx);  // RAII 上锁
    if (balance >= 500) balance -= 500;
}   // 自动解锁（无论异常还是正常返回）
```

**永远用 `lock_guard` / `unique_lock`，不要裸用 `lock()`/`unlock()`**——异常会跳过 `unlock()`，导致死锁。

### 死锁预防

```cpp
// 死锁：两个线程以相反顺序上锁
// 线程1：mtx_A → mtx_B
// 线程2：mtx_B → mtx_A

// 修复1：所有线程统一顺序
// 修复2：std::scoped_lock 原子地获取多把锁
std::scoped_lock guard(mtx_A, mtx_B);  // 顺序无关，内部防死锁
```

### 条件变量：等待某个条件成立

```cpp
std::condition_variable cv;
std::mutex mtx;
std::queue<int> tasks;

// 消费者
void consumer() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return !tasks.empty(); });
    // wait 内部：条件不满足 → 解锁+挂起 → 被notify唤醒 → 重新上锁+检查条件
    auto task = tasks.front(); tasks.pop();
}

// 生产者
void producer(int t) {
    { std::lock_guard<std::mutex> g(mtx); tasks.push(t); }
    cv.notify_one();
}
```

**虚假唤醒（Spurious Wakeup）**：操作系统可能无故唤醒线程，所以 `wait` 的 lambda 必须重新检查条件。

### CPU 条件变量 vs GPU 忙等

| | CPU `condition_variable` | GPU spin loop |
|--|--|--|
| 等待时 CPU 占用 | 0（线程挂起）| 100%（一直检查）|
| 唤醒延迟 | 较高（调度器介入）| 极低（下一周期）|
| 适用 | 等待时间长（毫秒级）| 等待时间短（微秒级）|

### 作业 Q&A

**Q1：`throw` 后 mutex 不解锁怎么办？**

抛出异常会跳过 `mtx.unlock()`，导致死锁。用 `lock_guard` —— RAII 保证析构时（栈展开时）自动 `unlock()`。

---

**Q2：`func_A` 锁 mtx1→mtx2，`func_B` 锁 mtx2→mtx1，会死锁吗？**

会。顺序相反，线程 1 持有 A 等 B，线程 2 持有 B 等 A，互相等待。
修复：统一上锁顺序，或用 `std::scoped_lock(mtx1, mtx2)`。

---

**Q3：持锁执行 `task()` 有什么问题？**

吞吐量：task 执行期间其他 worker 全部阻塞，N 线程退化成单线程。
最坏情况：task 内部调用 `queue.push()` → `push()` 尝试上同一把锁 → 自我死锁。
结论：临界区只保护队列操作（push/pop），执行 task 时必须先解锁。

---

## 第九课：RAII 与资源管理

### RAII 原则

**构造函数获取资源，析构函数释放资源。** C++ 保证对象离开作用域析构函数一定被调用（无论正常返回、提前 return 还是异常）。

```cpp
template<typename T>
class ScopedArray {
    T *ptr_;
public:
    explicit ScopedArray(size_t n) : ptr_(new T[n]) {}
    ~ScopedArray() { delete[] ptr_; }
    ScopedArray(const ScopedArray&) = delete;   // 禁止拷贝
};
```

### `unique_ptr`：独占所有权

```cpp
auto ptr = std::make_unique<int>(42);       // 替代 new int(42)
auto arr = std::make_unique<int[]>(1024);   // 替代 new int[1024]
auto ptr2 = std::move(ptr);   // 转移所有权，ptr 变 nullptr
// 离开作用域自动 delete
```

### `shared_ptr`：共享所有权

```cpp
auto a = std::make_shared<int>(100);
auto b = a;      // 引用计数 +1
// 最后一个 shared_ptr 析构时才 delete
```

**循环引用**：A 持有 B，B 持有 A → 引用计数永远不为 0 → 内存泄漏。
修复：把环中一条边改成 `weak_ptr`（不增加引用计数）。

### 移动语义

```cpp
// 拷贝：O(n)，两份数据并存
// 移动：O(1)，一份数据换主人
auto res = make_resource();  // 函数返回 unique_ptr：移动，不拷贝
```

**选择原则**：默认 `unique_ptr`（零开销）→ 真正需要共享才用 `shared_ptr` → 永远不用裸 `new/delete`。

### 作业 Q&A

**Q1：以下代码有几处资源泄漏？**
```cpp
int *buf = new int[512];
FILE *f = fopen(...);
if (!result.ok()) { delete[] buf; return; }   // ← f 泄漏
```
A 处 return：f 没有 fclose，**1 处泄漏**。用 RAII 两条路径都安全。

---

**Q2：循环引用 `a->next = b; b->next = a;`，"Node destroyed" 会打印吗？**

不会。a/b 局部变量析构后，引用计数从 2 降到 1（互相持有），永远不为 0，对象不释放，内存泄漏。
修复：`std::weak_ptr<Node> next`。

---

**Q3：`CudaMemory` 类设计**

```cpp
class CudaMemory {
    void *ptr_ = nullptr;
public:
    explicit CudaMemory(size_t bytes) { cudaMalloc(&ptr_, bytes); }
    ~CudaMemory() { if (ptr_) cudaFree(ptr_); }
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;
    CudaMemory(CudaMemory&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    CudaMemory& operator=(CudaMemory&& o) noexcept {
        if (this != &o) { if (ptr_) cudaFree(ptr_); ptr_ = o.ptr_; o.ptr_ = nullptr; }
        return *this;
    }
    void *data() const { return ptr_; }
};
```

移动构造必须将 `other.ptr_` 置 nullptr，防止析构时 double free。

---

## 第十课：迭代器模式

### 4 个必要操作

```cpp
*it          // 解引用：获取当前元素
++it         // 前进一步（前置）
it != end    // 终止判断
begin()/end() // 容器必须提供
```

只要实现这 4 个，就能用于 range-based for 和标准算法。

### range-based for 的展开

```cpp
for (int x : v) { ... }
// 等价于：
auto&& __range = v;
auto __begin = __range.begin();
auto __end   = __range.end();
for (; __begin != __end; ++__begin) { int x = *__begin; ... }
```

### 惰性迭代器：零内存分配

```cpp
// 生成 0, stride, 2*stride, ... 不存储任何数组
class StrideRange {
    class Iterator {
        int current_, stride_;
    public:
        int operator*() const { return current_; }
        Iterator& operator++() { current_ += stride_; return *this; }
        bool operator!=(const Iterator& o) const { return current_ < o.current_; }
        //  ↑ 用 < 而不是 !=，防止步长跨越终点导致无限循环
    };
    // ...
};
```

### `ExpertIterator` 的迭代器包装

```cpp
template<typename ExpertIteratorTy>
class ExpertRouteRange {
    class Iterator {
        ExpertIteratorTy& iter_;
        unsigned index_;
    public:
        ExpertAndOffset operator*() const { return iter_[index_]; }
        Iterator& operator++() { ++index_; return *this; }
        bool operator!=(const Iterator& o) const { return index_ != o.index_; }
    };
    // begin() → index=0，end() → index=K
};
// 使用：for (auto route : ExpertRouteRange(expert_iterator, K)) { ... }
```

### 作业 Q&A

**Q1：range-based for 展开形式**（见上方"range-based for 的展开"）

---

**Q2：`StrideRange(0, 7, 3)` 为什么会无限循环？**

序列 0→3→6→**9**→12→...，永远不会精确等于 7，`!=` 判断永远 true。
修复：用 `current_ < o.current_` 替代 `!=`。

---

**Q3：`ExpertRouteRange` 设计**（见上方代码）

---

## 第十一课：前缀和的三种实现

### 算法定义

```
输入：[3, 2, 5, 1]
inclusive：[3, 5, 10, 11]（含当前元素）
exclusive：[0, 3, 5, 10]（不含当前元素，首项为0）
```

### 三种实现对比

| | 串行 | Blelloch（CPU 并行）| Warp Shuffle（GPU）|
|--|--|--|--|
| 并行时间 | O(N) | O(log N) | O(log 32) = 5步 |
| 总工作量 | O(N) | O(N log N) | O(N)（N=32）|
| 内存访问 | 顺序 | shared memory | 零（寄存器）|
| 延迟 | 高 | 中 | 极低（4 cycle/轮）|

**Work-Efficiency 权衡**：Blelloch 总工作量是串行的 log N 倍，但并行时间是 log N 分之一——以工作量换时间。

### C++17 标准库

```cpp
std::inclusive_scan(a.begin(), a.end(), out.begin());
std::exclusive_scan(a.begin(), a.end(), out.begin(), 0 /*初始值*/);
std::inclusive_scan(std::execution::par, ...);  // 并行版本
```

### 惰性 ExclusiveScanRange

```cpp
class ExclusiveScanRange {
    class Iterator {
        const std::vector<int>& data_;
        int index_, running_;
    public:
        int operator*() const { return running_; }
        Iterator& operator++() { running_ += data_[index_++]; return *this; }
        bool operator!=(const Iterator& o) const { return index_ != o.index_; }
    };
public:
    Iterator begin() { return {data_, 0, 0}; }
    Iterator end()   { return {data_, (int)data_.size(), 0}; }
};
// 零内存分配，每步一个加法，结果在迭代器内部累积
```

### 作业 Q&A

**Q1：为什么串行前缀和无法并行？**

依赖链：`out[i]` 必须在 `out[i-1]` 算完之后才能算，形成长度 N-1 的串行链，任意两步之间都有直接依赖，无法并行。Blelloch 通过树形重排打破依赖链。

---

**Q2：`a2a_dispatch_send` 用的是 inclusive 还是 exclusive 语义？**

Inclusive。`expert_offsets[e]` = expert 0 到 e 的 token 总数（含 e）。
代码里用 `expert_offsets[expert - 1]` 找 expert 的起始位置，等价于 exclusive[expert]。

---

**Q3：`ExclusiveScanRange` 实现**（见上方代码）

---

## 第十二课：内存布局与缓存

### 行主序与缓存行

C++ 二维数组是**行主序（Row-Major）**：`matrix[i][0], [i][1], [i][2]...` 连续存储。

缓存行 64 字节（CPU）= 16 个 int，一次加载 16 个连续元素。

```
行优先遍历：步长 4 字节 → 每 16 次访问 1 次 miss → 缓存利用率 94%
列优先遍历：步长 N*4 字节 → 每次访问 1 次 miss → 缓存利用率 6%
                                                   → 慢 5-10 倍
```

### GPU Coalesced Access

Warp（32 线程）的内存请求被合并成尽量少的 128B 事务：

```cpp
// Coalesced ✅：相邻线程访问相邻地址
for (int i = threadIdx.x; i < N; i += blockDim.x)
    arr[i] = ...;   // 线程0→arr[0], 线程1→arr[1]...

// Non-coalesced ❌：步长 = blockDim.x，地址分散
arr[i * blockDim.x + threadIdx.x] = ...;
```

| 访问模式 | 内存事务数（32线程）| 带宽利用率 |
|--|--|--|
| 连续（coalesced）| 1-4 个 128B 事务 | ~100% |
| 步长 N（stride）| 最多 32 个事务 | 1/N |

### AoS vs SoA

```cpp
// AoS：struct 里有 x, y, z
// 只读 x：步长 = sizeof(struct) = 12，缓存利用率 4/12 ≈ 33%

// SoA：x[], y[], z[] 独立数组
// 只读 x：连续访问，利用率 100%
```

**GPU kernel 几乎全部用 SoA**——`a2a_dispatch_send` 里 `indices[]`、`weights[]`、`token_offset[]` 都是独立数组，而不是一个结构体数组，正是此原因。

### 对齐与 Padding

```cpp
struct Bad  { char a; double b; int c; };   // sizeof = 24（7+4 字节 padding）
struct Good { double b; int c; char a; };   // sizeof = 16（3 字节 padding）
// 规则：大字段放前面，减少 padding
```

`uint4`（16 字节，16 字节对齐）是 GPU 访问的理想单元，地址对齐后向量化读写生效——这是 `round_up(hidden_dim * elemsize, sizeof(int4))` 的最终原因。

### 作业 Q&A

**Q1：`struct A { char x; double y; int z; }` 的 sizeof 和内存布局**

```
偏移 0：x（1B）
偏移 1-7：padding（7B，double 需要 8 字节对齐）
偏移 8：y（8B）
偏移 16：z（4B）
偏移 20-23：padding（4B，struct 大小需是 8 的倍数）
sizeof = 24
```

优化：`{double y; int z; char x;}` → sizeof = 16。

---

**Q2：Phase 2 读 token 数据是 coalesced 访问吗？**

是。线程 i 读 `x_token_src[i]`，32 个线程读 32 个连续 `uint4`（每个 16B），共 512B，合并成 4 个 128B 内存事务，带宽利用率 100%。
前提：`x_token_src` 地址满足 16 字节对齐——正是 CPU 端 `round_up` 操作的保证。

---

**Q3：AoS 还是 SoA 对 GPU kernel 更友好？**

SoA（设计 B）。32 个线程读 `experts[0..31]`：连续地址，1 个内存事务，利用率 100%。
AoS：32 个线程读 `routes[0].expert, routes[1].expert, ...`：步长 12 字节（`sizeof(RouteInfo)`），32 个独立事务，利用率 33%。

---

## 课程总结

### 12 节课的知识图谱

```
生产代码 a2a_dispatch_send.cu
│
├─ 为什么 Fixed<N>？
│    └─ L4 模板特化：编译期选择实现版本
│         └─ L5 虚函数 vs 模板：vtable 开销，GPU 不能用虚函数
│              └─ L6 constexpr：整条优化链的起点
│
├─ 为什么需要 grid.sync()？
│    └─ L7 原子操作与内存序：happens-before，release-acquire
│         └─ L8 互斥锁与条件变量：临界区，死锁，条件等待
│
├─ 为什么 unique_ptr / RAII？
│    └─ L9 RAII 与资源管理：析构保证释放，移动语义零拷贝
│
├─ 为什么这样设计 ExpertIterator？
│    └─ L10 迭代器模式：抽象遍历逻辑，惰性求值
│         └─ L11 前缀和：同一算法在串行/CPU/GPU上的形态
│
└─ 为什么用独立数组而不是结构体数组？
     └─ L12 内存布局：Coalesced，SoA，对齐 → GPU 带宽利用率
```

### 一句话贯穿全课程

> 生产级 GPU kernel 的极致性能，来自**编译期决策（L4-L6）+ 正确同步（L7-L8）+ 零开销资源管理（L9）+ 高效数据结构（L10-L12）** 四层的协同设计。每一层都依赖其下层的保证，断开任何一层，性能或正确性都会崩塌。

---

*参考代码：[a2a_dispatch_send.cu](../pplx-garden/p2p-all-to-all/a2a-kernels/src/a2a/a2a_dispatch_send.cu)*
*CUDA 教学笔记：[cuda_three_lessons_complete.md](cuda_three_lessons_complete.md)*
