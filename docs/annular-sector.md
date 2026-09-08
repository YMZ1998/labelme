# Ring contour extraction and cutting



Choose **Ring** to use automatically estimated parameters without opening a

dialog. Choose **Ring Settings** to preview and adjust the extraction. The tool

fits the circular imaging field against the dark

background, then traces the inner dark-to-bright edge in polar coordinates.

Smoothness constraints join the inner boundary across noisy or shadowed angles.

The outer boundary is the imaging field, not an internal tissue boundary.



In **Ring Settings**, inspect the green contours. Adjust **Inner-edge position**

if the wrong radial

edge was selected and **Contour smoothness** to control local variation. Confirm,

then click outer start, inner start, inner end, and outer end. The fourth click

keeps the large arc; hold Shift for the small arc. The points define the cutting

sides. The remaining boundaries follow the extracted contour.

**Point spacing** controls the distance between generated polygon vertices in
image pixels. Its default comes from `[ring] point_spacing` in the project-root
`config.ini` (24 px in the provided file). Increase it for fewer points and easier
manual editing; decrease it when a detailed boundary needs closer tracking. Each
straight cutting side always contains exactly three points, including its two
endpoints.



The result is saved as an editable polygon. Undo the last point to adjust a side;

Escape cancels. Cancelling extraction does not change annotations. Changing the

image clears the extracted contour. The two cutting sides must intersect inside

the dark hole, and cannot be parallel or collinear. Results should be checked

against the image, especially around strong shadows or an unclear boundary.
