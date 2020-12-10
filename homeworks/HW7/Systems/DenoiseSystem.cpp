#include "DenoiseSystem.h"

#include "../Components/DenoiseData.h"

#include <_deps/imgui/imgui.h>

#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <Eigen/Sparse>

using namespace Ubpa;

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
						data->heMesh->Vertices().at(i)->idx = -1;
						data->heMesh->Vertices().at(i)->bidx = -1;
					}

					spdlog::info("Mesh to HEMesh success");
				}();
			}

			if (ImGui::Button("Solve Minimal surface")) {
				[&]() {
					if (!data->heMesh->IsTriMesh()) {
						spdlog::warn("HEMesh isn't triangle mesh");
						return;
					}

					const auto vertices = data->heMesh->Vertices();
					int totalPoints = static_cast<int>(vertices.size()), boundaryPoints = { 0 }, innerPoints = { 0 };
					for (int i = 0; i < totalPoints; ++i) {
						vertices[i]->idx = i;
						if (vertices[i]->IsOnBoundary()) ++boundaryPoints;
					}
					innerPoints = totalPoints - boundaryPoints;

					auto Idx = [&](int idx, int k) {
						return k * totalPoints + idx;
					};
					auto ConsIdx = [&](int idx, int k) {
						return 3 * totalPoints + idx * 3 + k;
					};

					Eigen::SparseMatrix<float> A(3 * (totalPoints + boundaryPoints), 3 * totalPoints);
					Eigen::SparseVector<float> b(3 * (totalPoints + boundaryPoints));
					spdlog::info(totalPoints);
					
					int b_idx = 0;
					for (auto* v : data->heMesh->Vertices()) {
						const auto P = v->position;

						// A_ij & x_i
						if (v->IsOnBoundary()) {
							for (auto k : { 0, 1, 2 }) {
								b.coeffRef(ConsIdx(b_idx, k)) = P[k];
								A.coeffRef(ConsIdx(b_idx, k), Idx(v->idx, k)) = 1;
							}
							++b_idx;
						}
						const auto& adj_v = v->AdjVertices();
						for (auto k : { 0, 1, 2 }) 
							A.coeffRef(Idx(v->idx, k), Idx(v->idx, k)) = v->AdjVertices().size();
							
						for (auto* adj : adj_v) {
							for (auto k : { 0, 1, 2 }) A.coeffRef(Idx(v->idx, k), Idx(adj->idx, k)) = -1;
						}
					}
					spdlog::info("Build Matrix successful");

					Eigen::ConjugateGradient<Eigen::SparseMatrix<float> > solver;
					solver.setTolerance(1e-4);
					solver.compute(A.transpose() * A);
					Eigen::SparseVector<float> x(3 * totalPoints);
					x = solver.solve(A.transpose() * b);

					spdlog::info("Solve Matrix successful");

					for (auto* v : data->heMesh->Vertices()) {
						for (auto k : { 0, 1, 2 }) {
							v->position[k] = x.coeffRef(Idx(v->idx, k));
						}
					}
					spdlog::info("Data transform successful");
				}();
			}

			if (ImGui::Button("Parameterization")) {
				[&]() {
					if (!data->heMesh->IsTriMesh()) {
						spdlog::warn("HEMesh isn't triangle mesh");
						return;
					}

					const auto vertices = data->heMesh->Vertices();
					int totalPoints = static_cast<int>(vertices.size()), boundaryPoints = { 0 }, innerPoints = { 0 };
					for (int i = 0; i < totalPoints; ++i) vertices[i]->idx = i;
					int bpt;
					for (int i = 0; i < totalPoints; ++i)
						if (vertices[i]->IsOnBoundary()) bpt = vertices[i]->idx; // start point
					do {
						vertices[bpt]->bidx = boundaryPoints++;
						const auto& adj_v = vertices[bpt]->AdjVertices();
						for (auto* v : data->heMesh->Vertices()) {
							if (v->IsOnBoundary() && v->bidx < 0) {
								bpt = v->idx;
								break;
							}
						}
					} while (vertices[bpt]->bidx < 0);

					innerPoints = totalPoints - boundaryPoints;

					auto Idx = [&](int idx, int k) {
						return k * totalPoints + idx;
					};
					auto ConsIdx = [&](int idx, int k) {
						return 2 * totalPoints + idx * 2 + k;
					};

					Eigen::SparseMatrix<float> A(2 * (totalPoints + boundaryPoints), 2 * totalPoints);
					Eigen::SparseVector<float> b(2 * (totalPoints + boundaryPoints));
					spdlog::info(totalPoints);

					for (auto* v : data->heMesh->Vertices()) {
						const auto P = v->position;

						// A_ij & x_i
						if (v->IsOnBoundary()) {
							{ // project to UV surface
								float t = 4 * float(v->bidx) / float(boundaryPoints);
								float U = { 0.0f }, V = { 0.0f };
								if (t <= 1.0f) V = t;
								else if (t <= 2.0f) U = t - 1, V = 1.0f;
								else if (t <= 3.0f) U = 1.0f, V = 1.0f - (t - 2.0f);
								else U = 1.0f - (t - 3.0f);

								spdlog::info(std::to_string(U) + " " + std::to_string(V));

								b.coeffRef(ConsIdx(v->bidx, 0)) = U;
								b.coeffRef(ConsIdx(v->bidx, 1)) = V;
							}
							for (auto k : { 0, 1 }) A.coeffRef(ConsIdx(v->bidx, k), Idx(v->idx, k)) = 1;
						}
						const auto& adj_v = v->AdjVertices();
						for (auto k : { 0, 1 })
							A.coeffRef(Idx(v->idx, k), Idx(v->idx, k)) = v->AdjVertices().size();

						for (auto* adj : adj_v) {
							for (auto k : { 0, 1 }) A.coeffRef(Idx(v->idx, k), Idx(adj->idx, k)) = -1;
						}
					}
					spdlog::info("Build Matrix successful");

					Eigen::ConjugateGradient<Eigen::SparseMatrix<float> > solver;
					solver.setTolerance(1e-2);
					solver.compute(A.transpose() * A);
					Eigen::SparseVector<float> x(2 * totalPoints);
					x = solver.solve(A.transpose() * b);

					spdlog::info("Solve Matrix successful");

					for (auto* v : data->heMesh->Vertices()) {
						v->uv = { x.coeffRef(Idx(v->idx, 0)), x.coeffRef(Idx(v->idx, 1)) };
						spdlog::info(std::to_string(v->uv[0]) + " " + std::to_string(v->uv[1]));
					}

					spdlog::info("Data transform successful");
				}();
			}

			if (ImGui::Button("Set Normal to Color")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					data->mesh->SetToEditable();
					const auto& normals = data->mesh->GetNormals();
					std::vector<rgbf> colors;
					for (const auto& n : normals)
						colors.push_back((n.as<valf3>() + valf3{ 1.f }) / 2.f);
					data->mesh->SetColors(std::move(colors));

					spdlog::info("Set Normal to Color Success");
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
					std::vector<Ubpa::pointf3> positions(N);
					std::vector<Ubpa::pointf2> uvs(N);
					std::vector<uint32_t> indices(M * 3);
					for (size_t i = 0; i < N; i++) {
						positions[i] = data->heMesh->Vertices().at(i)->position;
						uvs[i] = data->heMesh->Vertices().at(i)->uv;
					}
					for (size_t i = 0; i < M; i++) {
						auto tri = data->heMesh->Indices(data->heMesh->Polygons().at(i));
						indices[3 * i + 0] = static_cast<uint32_t>(tri[0]);
						indices[3 * i + 1] = static_cast<uint32_t>(tri[1]);
						indices[3 * i + 2] = static_cast<uint32_t>(tri[2]);
					}
					data->mesh->SetPositions(std::move(positions));
					data->mesh->SetUV(std::move(uvs));
					data->mesh->SetIndices(std::move(indices));
					data->mesh->SetSubMeshCount(1);
					data->mesh->SetSubMesh(0, { 0, M * 3 });
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
