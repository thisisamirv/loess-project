# Online Adapter

Incremental updates with a sliding window for real-time data.

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](online_comparison.svg)

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `window_capacity` | 1000 | Max points in window |
| `min_points` | 2 | Points before output starts |
| `update_mode` | `"incremental"` | Update strategy |

## Update Modes

| Mode | Behavior | Speed |
| --- | --- | --- |
| `"incremental"` | Update only affected fits | Faster |
| `"full"` | Recompute entire window | More accurate |

## Example

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastloess::OnlineOptions opts;
    opts.fraction = 0.2;
    opts.iterations = 1;
    opts.window_capacity = 100;
    opts.min_points = 5;
    opts.update_mode = "incremental";

    fastloess::OnlineLoess model(opts);
    for (size_t i = 0; i < x.size(); ++i) {
        auto out = model.add_point(x[i], y[i]).value();
        if (out.has_value())
            std::cout << out.y() << std::endl;
    }

    return 0;
}
```

```output
0.351148
0.412033
0.471662
0.529795
0.586197
0.640641
0.692908
0.742788
0.790079
0.834592
0.876146
0.914576
0.949725
0.981453
1.00963
1.02713
1.04896
1.06697
1.08109
1.09125
1.09743
1.09959
1.09772
1.09184
1.08196
1.07028
1.05321
1.03231
1.00766
0.979345
0.947493
0.912229
0.873694
0.832044
0.787446
0.747569
0.698065
0.646154
0.592043
0.535951
0.478103
0.418733
0.35808
0.299623
0.237194
0.181014
0.117788
0.0544907
-0.00862337
-0.0726687
-0.129706
-0.190789
-0.250701
-0.309201
-0.366053
-0.415095
-0.468166
-0.51895
-0.567242
-0.612846
-0.655581
-0.695273
-0.731762
-0.762596
-0.79245
-0.815586
-0.838505
-0.857644
-0.872928
-0.883746
-0.890791
-0.894383
-0.893971
-0.889556
-0.881157
-0.869352
-0.853612
-0.834032
-0.810691
-0.783683
-0.754041
-0.720168
-0.682992
-0.642663
-0.599344
-0.55782
-0.509371
-0.458469
-0.405317
-0.350766
-0.295792
-0.237339
-0.177529
-0.116601
-0.054801
5.06808e-05
```

---
