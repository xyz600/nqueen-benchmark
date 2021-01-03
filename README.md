# nqueen-benchmark

compare gpu & cpu n-queen performance

## 基本実装

* 枝刈りありの backtrack
* 1 step で 1行ずつ埋めていく
* 縦方向と斜め2方向に関して、bitmap を使って利きを判定
    * `n <= 32` を前提にするなら、多くても 64 bit の bitmap があれば十分

## CPU implementation

### ver1

単純な実装  

結果(N = 8 to 15)

```
result count: 92, elapsed = 0[ms]
result count: 352, elapsed = 0[ms]
result count: 724, elapsed = 2[ms]
result count: 2680, elapsed = 10[ms]
result count: 14200, elapsed = 34[ms]
result count: 73712, elapsed = 193[ms]
result count: 365596, elapsed = 1162[ms]
result count: 2279184, elapsed = 7451[ms]
```

### ver2

Open MP による並列化  

結果(N = 8 to 17)

```
result count: 92, elapsed = 13[ms]
result count: 352, elapsed = 1[ms]
result count: 724, elapsed = 2[ms]
result count: 2680, elapsed = 1[ms]
result count: 14200, elapsed = 3[ms]
result count: 73712, elapsed = 19[ms]
result count: 365596, elapsed = 119[ms]
result count: 2279184, elapsed = 689[ms]
result count: 14772512, elapsed = 4405[ms]
result count: 95815104, elapsed = 29532[ms]
```

### ver3

深さ2まで展開して、並列度を調整  

結果(N = 8 to 17)

```
result count: 92, elapsed = 8[ms]
result count: 352, elapsed = 0[ms]
result count: 724, elapsed = 0[ms]
result count: 2680, elapsed = 0[ms]
result count: 14200, elapsed = 4[ms]
result count: 73712, elapsed = 21[ms]
result count: 365596, elapsed = 80[ms]
result count: 2279184, elapsed = 557[ms]
result count: 14772512, elapsed = 3686[ms]
result count: 95815104, elapsed = 24117[ms]
```

### ver4

最内ループを SIMD 4並列

結果(N = 8 to 17)

```
result count: 92, elapsed = 3[ms]
result count: 352, elapsed = 13[ms]
result count: 724, elapsed = 2[ms]
result count: 2680, elapsed = 4[ms]
result count: 14200, elapsed = 1[ms]
result count: 73712, elapsed = 9[ms]
result count: 365596, elapsed = 67[ms]
result count: 2279184, elapsed = 366[ms]
result count: 14772512, elapsed = 2155[ms]
result count: 95815104, elapsed = 18434[ms]
```