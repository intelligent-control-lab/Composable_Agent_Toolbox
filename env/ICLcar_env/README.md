# How to make your own track

1. Tracks are represented by binary images in this environment. To create tracks, the easiest way is through GIMP. You'll want to first install the GIMP Image Editing Software.

2. Create a new project and fill the background to be black (or whatever color that represents the region not the track).

3. Press the B key to use the path tool. Start clicking points on the canvas to form a curve. Ctrl + click on the initial point to close the curve. You can adjust the curvature of the curve by clicking and dragging the edges.

4. On the side panel, click on stroke path. This will give your path some width. Adjust the color and line width accordingly.

5. Stroke again with a smaller line width and dash pattern to create the center lane markings.

6. Feel free to add other miscellaneous objects on the road via the shape tool.

7. When you are done creating the track, click on File -> Export As -> Select BMP as the file type -> Save.


# Important

* Split up your road into several layers and export each layer as its own image to make it easier to determine the binary masks for object collision. The layers we need are:
    * (a) the road
    * (b) textures for different friction level (these are ideally split into layers
    * (c) obstacles (optional)
    * (d): starting location (optional)

* See image below for an example
![example](../misc/example.png)
![example_layers](../misc/example_layers.png)


## Notes

* To add a checker pattern (for the start location): Filters -> Render -> Pattern -> Checkboard ...

* Bucket fill with pattern to create different textures on the road.