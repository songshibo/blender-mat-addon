# blender-mat-addon

(Import-Only for now, working on exporting). Pls refer to [link](https://songshibo.github.io/2021/04/12/Medial-Axis-Transform-Mesh-Generation/) for detailed generation process.

This addon will generate medial meshes & interpolated MATs from MA files in blender

MA file(Different from [mayAscii](https://download.autodesk.com/us/maya/2011help/index.html?url=./files/Maya_ASCII_file_format.htm,topicNumber=d0e702047)) stores the the information of medial mesh. It can be generated using [Q-MAT](http://cgcad.thss.tsinghua.edu.cn/wangbin/qmat/qmat.html) or [Q-MAT+](https://personal.utdallas.edu/~xguo/GMP2019.pdf).

## Usage

- Clone this repo.
- Open blender.
- Edit > Preferences > Addons.
- Click Install button.
- Select ```mat_import/__init__.py``` from this repo.
- Check the checkbox of the MAT add-on to enable it.

## Updates

- Improve the performance of importing high-resolution medial mesh
- Use icosphere instead of UV sphere
- Add subdivision of icosphere/ initial radius of medial sphere
- Using blender-vscode plugin for development.

### Update(2021-10-09)

- Add a new generation mode: each medial primtive will be generated as a single object and named by its index.

#### Usage:

- Simply select "Individual Primitive" to switch generation mode.

<img src="./render_results/primitive mode.png" alt="primitive mode" style="zoom:30%;" />

#### Results

In primitive generation mode, each medial primitive will be created as a individual object containing medial sphere/conical surface of medial cone/slab.

<img src="./render_results/PMResult.png" alt="PMResult" style="zoom:40%;" />

## Results

this addon will import medial axis transform as several objects:

- Medial mesh: named as ```${filename}```.
- Medial sphere: named as its vertex number. eg ```v1```. Its global position and scale are the center & radius of the sphere respectively. All spheres are packed into a group named ```${filename}.SphereGroup```.
- Medial cone: single mesh object named ```${filename}.ConeGroup```.
- Medial slab: single mesh object named ```${filename}.SlabGroup```.

| <img src=".\render_results\medial mesh.png" alt="medial mesh" style="zoom:33%;" /> | <img src=".\render_results\sphere.png" alt="sphere" style="zoom:33%;" /> | <img src=".\render_results\cone.png" alt="cone" style="zoom:33%;" /> | <img src=".\render_results\slab.png" alt="slab" style="zoom:33%;" /> | <img src=".\render_results\result.png" alt="result" style="zoom:33%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         medial mesh                          |                        medial spheres                        |                         medial cones                         |                         medial slabs                         |                           combined                           |



 ## Requirements

- Blender 2.80.0 or older

## MA file structure

> /# number of vertex(medial sphere)/edge(medial cone)/face(medial slab)
>
> vertices edges faces
>
> \# v/e/f indicates the type represented by current line
>
> /# (x,y,z): center of the medial sphere; r: radius
>
> v x y z r
>
> /# two end vertices of the edge
>
> e v1 v2
>
> /# three vertices of a triangle face
>
> f v3 v4 v5	
>
> \#  comment lines in MA file should start with #

