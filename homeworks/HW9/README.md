# 作业9报告

2020/12/26 余畅 电子科技大学

### 问题

+ 实现网格简化的QEM方法 
	+ Garland and Heckbert. Surface Simplification Using Quadric Error Metrics. Siggraph 1997. 
	+ http://mgarland.org/files/papers/quadrics.pdf 
+  学习网格的拓扑关系（点、边、面）更新的维护

### 实现

#### Quadric

原论文Quadric Error Metric中的 `Quadric` $Q$，定义为：

$$Q=（\mathbf{A},\mathbf{b},c)=(\mathbf{n}\mathbf{n}^{T}, d\mathbf{n}, d^2)$$，其中 $\mathbf{n}$ 为某个平面的法线，$d$ 是 $ax+by+cz+d=0$ 中的 $d$。

所以点到平面的平方距离就可以按照下式求得：

$Q(\mathbf{v})=\mathbf{v}^T\mathbf{A}\mathbf{v}+2\mathbf{b}^T\mathbf{v}+c$

度量一个点移动之后产生的 error，就可以用 $\sum_i Q_i(\mathbf{v})$ 来计算，其中 $i$ 是和点 $\mathbf{v}$ 相邻的平面。

经过推导，可以用以下代码来计算 $Q$ ：

```C++
const auto& adj = f->AdjVertices();

valf3 v1 = adj[0]->position;
valf3 v2 = adj[1]->position;
valf3 v3 = adj[2]->position;

Vector4f p;
/* http://paulbourke.net/geometry/planeeq/ */
p[0] = v1[1] * (v2[2] - v3[2]) + v2[1] * (v3[2] - v1[2]) + v3[1] * (v1[2] - v2[2]);
p[1] = v1[2] * (v2[0] - v3[0]) + v2[2] * (v3[0] - v1[0]) + v3[2] * (v1[0] - v2[0]);
p[2] = v1[0] * (v2[1] - v3[1]) + v2[0] * (v3[1] - v1[1]) + v3[0] * (v1[1] - v2[1]);
p[3] = -(v1[0] * (v2[1] * v3[2] - v3[1] * v2[2]) +
		v2[0] * (v3[1] * v1[2] - v1[1] * v3[2]) +
		v3[0] * (v1[1] * v2[2] - v2[1] * v1[2]));

Matrix4f pp = p * p.transpose();
```



#### 算法流程

为了对曲面进行修改，我们需要临时的数据结果来储存点-面的关系，在给每个点和面唯一的 `id` 之后，`verteces` 保存了 `id->Vertex` 的映射，`vertexToFaces` 可通过点 `id` 查找邻接面，`data->faces` 可通过面 `id` 查找邻接点，`initialQ` 保存了每个点的 `Quadric` $Q$ 。 `edgeset` 则按照 <id,id> 的键值对保存了所有的边。对应的数据定义如下：

```C++
// DenoiseData.h
[[UInspector::hide]]
std::vector<std::vector<Vertex*> > faces;

// DenoiseSystem.cpp
std::vector<Vertex*> verteces(total + 1, NULL);
std::vector<Matrix4f> initialQ(total + 1, Matrix4f::Zero());
std::vector<std::vector<Triangle*> > vertexToFaces(total + 1, std::vector<Triangle*>());

std::set<vertpair> edgeset;
```



初始化的代码如下：

```C++
for (int i = 0; i < data->heMesh->Polygons().size(); i++) {
    
    //...
    
	for (auto vert : f->AdjVertices()) {
		verteces[vert->id] = vert;
		vertexToFaces[vert->id].push_back(f);
		data->faces[f->id].push_back(vert);

		initialQ[vert->id] += pp;
	}

	for (const auto& edge : f->AdjEdges()) {
		std::vector<Vertex*> tmpV;
		for (const auto& vert : edge->AdjVertices()) {
			tmpV.push_back(vert);
		}
		vertpair vp = vertpair(tmpV[0]->id, tmpV[1]->id);
		edgeset.insert(vp);
	}
    
    //...
    
}
```



我们使用 `Contr` 类来描述将两个点（即一条边）合并成一个点的操作。将所有的初始边加入优先队列。每次从中取出 $Q$ 值最小的点更新，由于合并完一个点之后，队列中的其余的边也会改变，所以实现的还是 $O(N^2)$ 的时间复杂度。相应代码如下：

```C++
std::deque<Contr> edges;
for (auto edge = edgeset.begin(); edge != edgeset.end(); edge++) {
	edges.push_back(Contr(*edge, verteces, initialQ));
}

data->facesToRemove.clear();
data->verticesToRemove.clear();

int target_edges = (int)(data->scale * (float)edges.size());
while (edges.size() > target_edges) {
	Contr best = popmin(edges);
	best.perform(verteces, initialQ, vertexToFaces, data->facesToRemove, data->verticesToRemove, edges, edgeset, data->faces);
}
```



优先队列的比较：

```C++
bool operator<(const Contr& other) const {
	return resultError < other.resultError;
}
```



#### 合并操作

对于合并的操作，处于方便，我们考虑三种类型：合并到边的左端点、右端点或者中点。取三者中的 $Q$ 的最小值：

```C++
enum collapse_method {
	COLLAPSE_TO_V1,
	COLLAPSE_TO_V2,
	COLLAPSE_TO_MEAN
};

// ...

class Contr {
public:
	vertpair vp;
	Matrix4f Q;
	Vector4f loc;
	collapse_method method;
	float resultError;
    
    // ...
    
   void findMinError(const std::vector<Vertex*>& verteces,
	const std::vector<Matrix4f>& initialQ) {
	Vector4f loc_v1 = homogenous(verteces[vp.first]->position);
	Vector4f loc_v2 = homogenous(verteces[vp.second]->position);
	Vector4f loc_vm = (loc_v1 + loc_v2) / 2;

	Q = initialQ[vp.first] + initialQ[vp.second];

	float err_v1 = error(loc_v1, Q),
		err_v2 = error(loc_v2, Q),
		err_vm = error(loc_vm, Q);


	if (err_v1 < err_v2) {
		if (err_vm < err_v1) method = COLLAPSE_TO_MEAN;
		else method = COLLAPSE_TO_V1;
	}
	else {
		if (err_vm < err_v2) method = COLLAPSE_TO_MEAN;
		else method = COLLAPSE_TO_V2;
	}

	if (method == COLLAPSE_TO_V1) {
		loc = loc_v1;
		resultError = err_v1;
	}
	else if (method == COLLAPSE_TO_V2) {
		loc = loc_v2;
		resultError = err_v2;
	}
	else {
		loc = loc_vm;
		resultError = err_vm;
	}
}
```



合并操作的时候，首先更新合并后顶点的位置：

```C++
// perform() ...
initialQ[vp.first] = Q;

int keep = vp.first, remove = vp.second;
if (method == COLLAPSE_TO_MEAN) {
	valf3 tmp = verteces[keep]->position;
	tmp += verteces[remove]->position;
	tmp /= 2;
	verteces[keep]->position = tmp;
}
else if (method == COLLAPSE_TO_V2) {
	verteces[keep]->position = verteces[remove]->position;
}
// ...
```



然后给删除的点打上 tag，并遍历删除点的邻接面。如果邻接面中包含另一个点（即包含正在进行合并的边），那么这个面的面积变成 0，给这个面打上删除tag，否则将这个面中删除的顶点更新为另一个点：

```C++
// perform() ...
std::vector<Triangle*> faces = vertexToFaces[remove];
for (int i = 0; i < faces.size(); i++) {
	Triangle* f = faces[i];

	int v1idx = -1, v2idx = -1;
	for (int j = 0; j < faceAdjVert[f->id].size(); j++) {
		if (faceAdjVert[f->id][j]->id == keep) v1idx = j;
		else if (faceAdjVert[f->id][j]->id == remove) v2idx = j;
	}

	verticesToRemove.insert(remove);
	faceAdjVert[f->id][v2idx] = verteces[keep];
	if (v1idx == -1) {
		vertexToFaces[keep].push_back(f);
	}
	else {
		facesToRemove.insert(f->id);
	}
}
//...
```



最后更新优先队列，对存在删除点的边进行更新：

```C++
// perform() ...
std::deque<int> edgesToRemove;
for (int i = 0; i < edges.size(); i++) {
	if (vp == edges[i].vp) continue;
	if (edges[i].contains(keep)) {
		edges[i].findMinError(verteces, initialQ);
	}
	else if (edges[i].contains(remove)) {
		vertpair possible;
		if (edges[i].vp.first == remove) {
			possible = std::make_pair(edges[i].vp.second, keep);
		}
		else {
			possible = std::make_pair(edges[i].vp.first, keep);
		}

		if (existingedges.find(possible) != existingedges.end()) {
			edgesToRemove.push_back(i);
		}
		else {
			edges[i].vp = possible;
			edges[i].findMinError(verteces, initialQ);
		}
	}
}

int offset = 0;
for (int i = 0; i < edgesToRemove.size(); i++) {
	existingedges.erase(edges[edgesToRemove[i]].vp);
	edges.erase(edges.begin() + edgesToRemove[i] + offset);
	offset--;
}

vertexToFaces[remove].clear();
```



#### HeMesh To Mesh

最终需要将 HeMesh 转为 Mesh，待删除的点和面分别储存在 `facesToRemove` 和 `verticesToRemove` 的set中。在加入点和面的时候分别在set里面查找，如果没有则加入：

```C++
if (ImGui::Button("HEMesh to Mesh")) {
	[&]() {
		if (!data->mesh) {
			spdlog::warn("mesh is nullptr");
			return;
		}

		if (!data->heMesh->IsTriMesh() || data->heMesh->IsEmpty()) {
			spdlog::warn("HEMesh isn't triangle mesh or is empty");
			return;
		}

		data->mesh->SetToEditable();

		const size_t N = data->heMesh->Vertices().size();
		const size_t M = data->heMesh->Polygons().size();
		std::vector<Ubpa::pointf3> positions;
		std::vector<uint32_t> indices;
		std::vector<int> remapping(N + 1, -1);
		for (size_t i = 0; i < N; i++) {
			int id = data->heMesh->Vertices().at(i)->id;
			//if (data->verticesToRemove.find(id) == data->verticesToRemove.end()) {
			positions.push_back(data->heMesh->Vertices().at(i)->position);
			remapping[id] = static_cast<int>(positions.size() - 1);
			//}
		}
		for (size_t i = 0; i < M; i++) {
			int id = i;
			if (data->facesToRemove.find(id) == data->facesToRemove.end()) {
				/*if (remapping[data->faces[id][0]->id] == -1 ||
					remapping[data->faces[id][1]->id] == -1 ||
					remapping[data->faces[id][2]->id] == -1) continue;*/
				indices.push_back(static_cast<uint32_t>(remapping[data->faces[id][0]->id]));
				indices.push_back(static_cast<uint32_t>(remapping[data->faces[id][1]->id]));
				indices.push_back(static_cast<uint32_t>(remapping[data->faces[id][2]->id]));
			}
		}
		const size_t M3 = indices.size();
		data->mesh->SetColors({});
		data->mesh->SetUV({});
		data->mesh->SetNormals({});
		data->mesh->SetPositions(std::move(positions));
		data->mesh->SetIndices(std::move(indices));
		data->mesh->SetSubMeshCount(1);
		data->mesh->SetSubMesh(0, { 0, M3 });
		data->mesh->GenUV();
		data->mesh->GenNormals();
		data->mesh->GenTangents();

		spdlog::info("HEMesh to Mesh success");
	}();
}
```



### 结果

实现结果见 `./Videos` 文件夹下的演示视频。



g1n0st

2020/12/26