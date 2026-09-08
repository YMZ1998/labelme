# Strip annotations

Choose **Strip**, then click along the centerline of a thin region. Use extra
points where the target bends. Press Enter or Space, double-click, or
Ctrl/Cmd+click the last point to finish. Labelme expands the centerline into a
closed, editable polygon.

Set the width in the project-root `config.ini`:

```ini
[strip]
half_width = 8
```

`half_width` is measured in image pixels and accepts values from 1 through 100.
The resulting full width is approximately twice this value. The first version
uses a constant width along the centerline; edit the resulting polygon when the
target width varies.
