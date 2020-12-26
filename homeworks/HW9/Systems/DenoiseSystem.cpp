#include "DenoiseSystem.h"

#include "../Components/DenoiseData.h"

#include <_deps/imgui/imgui.h>

#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <vector>
#include <deque>
#include <set>

using namespace Eigen;
using namespace Ubpa;

using vertpair = std::pair<int, int>;

enum collapse_method {
	COLLAPSE_TO_V1,
	COLLAPSE_TO_V2,
	COLLAPSE_TO_MEAN
};

inline Vector4f homogenous(const valf3& vec) {
	return Vector4f(vec[0], vec[1], vec[2], 1);
}

float error(const Vector4f& homo, const Matrix4f& Q) {
	return abs((float)(homo.transpose() * Q * homo));
}

class Contr {
public:
	vertpair vp;
	Matrix4f Q;
	Vector4f loc;
	collapse_method method;
	float resultError;

	Contr(
		const vertpair& vertp, const std::vector<Vertex*>& verteces,
		const std::vector<Matrix4f>& initialQ) {
		vp = vertp;
		findMinError(verteces, initialQ);
	}

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

	bool operator<(const Contr& other) const {
		return resultError < other.resultError;
	}

	bool contains(int vid) {
		return vid == vp.first || vid == vp.second;
	}

	void perform(std::vector<Vertex*>& verteces, std::vector<Matrix4f>& initialQ,
		std::vector<std::vector<Triangle*> >& vertexToFaces,
		std::set<int>& facesToRemove,
		std::set<int>& verticesToRemove,
		std::deque<Contr>& edges,
		std::set<vertpair>& existingedges,
		std::vector<std::vector<Vertex*> >&
		faceAdjVert) {

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
	}
};

Contr popmin(std::deque<Contr>& edges) {
	Contr best = edges.front();
	int bestidx = 0;
	for (int i = 1; i < edges.size(); i++) {
		if (edges[i] < best) {
			bestidx = i;
			best = edges[i];
		}
	}
	edges[bestidx] = edges.back();
	edges.pop_back();
	return best;
}

void DenoiseSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<DenoiseData>();
		if (!data)
			return;

		if (ImGui::Begin("Denoise")) {
			if (ImGui::Button("Mesh to HEMesh")) {
				data->heMesh->Clear();
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					if (data->mesh->GetSubMeshes().size() != 1) {
						spdlog::warn("number of submeshes isn't 1");
						return;
					}

					data->copy = *data->mesh;

					std::vector<size_t> indices(data->mesh->GetIndices().begin(), data->mesh->GetIndices().end());

					data->heMesh->Init(indices, 3);
					if (!data->heMesh->IsTriMesh())
						spdlog::warn("HEMesh init fail");
					
					for (size_t i = 0; i < data->mesh->GetPositions().size(); i++) {
						data->heMesh->Vertices().at(i)->position = data->mesh->GetPositions().at(i);
						data->heMesh->Vertices().at(i)->id = -1;
					}

					spdlog::info("Mesh to HEMesh success");
				}();
			}

			if (ImGui::Button("QEM")) {
				[&]() {
					if (!data->heMesh->IsTriMesh()) {
						spdlog::warn("HEMesh isn't triangle mesh");
						return;
					}

					const auto vertices = data->heMesh->Vertices();
					auto total = vertices.size();
					for (int i = 0; i < total; ++i) vertices[i]->id = i;

					data->faces.clear();
					data->faces.resize(data->heMesh->Polygons().size() + 1, std::vector<Vertex*>());

					std::vector<Vertex*> verteces(total + 1, NULL);
					std::vector<Matrix4f> initialQ(total + 1, Matrix4f::Zero());
					std::vector<std::vector<Triangle*> > vertexToFaces(total + 1, std::vector<Triangle*>());

					std::set<vertpair> edgeset;

					for (int i = 0; i < data->heMesh->Polygons().size(); i++) {
						Triangle* f = data->heMesh->Polygons()[i];
						f->id = i;

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
					}

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
				}();
			}

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

			if (ImGui::Button("Recover Mesh")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					if (data->copy.GetPositions().empty()) {
						spdlog::warn("copied mesh is empty");
						return;
					}

					*data->mesh = data->copy;

					spdlog::info("recover success");
				}();
			}
		}
		ImGui::End();
	});
}
