# blender-mat-addon
This addon will generate medial meshes & interpolated MATs from MA files in blender

MA file(Different from [mayAscii](https://download.autodesk.com/us/maya/2011help/index.html?url=./files/Maya_ASCII_file_format.htm,topicNumber=d0e702047)) stores the the information of medial mesh. It can be generated using [Q-MAT](http://cgcad.thss.tsinghua.edu.cn/wangbin/qmat/qmat.html) or [Q-MAT+](https://personal.utdallas.edu/~xguo/GMP2019.pdf).

### MA file structure

> vertices edges faces	// number of vertex(medial sphere)/edge(medial cone)/face(medial slab)
>
> // v/e/f indicates the type represented by current line
>
> v x y z r	// (x,y,z): center of the medial sphere; r: radius
>
> e v1 v2	// two end vertices of the edge
>
> f v3 v4 v5	// three vertices of a triangle face
>
> // MA file should contain no annotation