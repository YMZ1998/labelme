# ROI tool defaults

Polygon, strip, and ring drawing use tool-specific default labels and shortcuts.
Configure them in the project-root `config.ini`:

```ini
[labels]
polygon = 1
strip = 2
ring = 5

[shortcuts]
create_strip = Ctrl+Q
create_ring = Ctrl+W
create_polygon = Ctrl+E
```

An ROI can be selected while a drawing tool is active by holding Shift and
left-clicking it. Shift+left-click on empty space continues to perform the
active tool's normal drawing behavior.
