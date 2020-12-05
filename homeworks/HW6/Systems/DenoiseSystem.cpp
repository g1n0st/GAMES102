#include "DenoiseSystem.h"

#include "../Components/DenoiseData.h"

#include <_deps/imgui/imgui.h>

#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>

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

					const auto& normals = data->mesh->GetNormals();
					assert(normals.size() == data->heMesh->Vertices().size());

					if (!data->heMesh->IsTriMesh())
						spdlog::warn("HEMesh init fail");
					
					for (size_t i = 0; i < data->mesh->GetPositions().size(); i++) {
						data->heMesh->Vertices().at(i)->position = data->mesh->GetPositions().at(i);
						data->heMesh->Vertices().at(i)->normal = normals[i];
					}

					spdlog::info("Mesh to HEMesh success");
				}();
			}

			if (ImGui::Button("Process")) {
				[&]() {
					if (!data->heMesh->IsTriMesh()) {
						spdlog::warn("HEMesh isn't triangle mesh");
						return;
					}
					
					for (int k = 0; k < data->iter_time; ++k) {
						for (auto* v : data->heMesh->Vertices())
							v->old_position = v->position; // backup old postion

						for (auto* v : data->heMesh->Vertices()) {
							if (v->IsOnBoundary()) continue; // fix the boundary points
						
							const auto P = v->old_position;
							valf3 Hn = { 0.f };

							float A = 0.f;
							const auto& adj_a = v->AdjPolygons();
							for (auto* poly : adj_a) {
								A += poly->Area();
							}
							spdlog::info(std::to_string(A));
							assert(A > 0.f);

							const auto& adj_v = v->AdjVertices();
							for (size_t i = 0; i < adj_v.size(); ++i) {
								const auto p0 = (i != 0 ? adj_v[i - 1]->old_position : adj_v.back()->old_position);
								const auto p1 = adj_v[i]->old_position;
								const auto p2 = (i + 1 < adj_v.size() ? adj_v[i + 1]->old_position : adj_v[0]->old_position);
								float cos_a = (p1 - p0).dot(P - p0);
								float cos_b = (p1 - p2).dot(P - p2);
								cos_a = std::min(cos_a, 0.9f);
								cos_b = std::min(cos_b, 0.9f);
								float cot_a = cos_a / sqrt(1 - cos_a * cos_a);
								float cot_b = cos_b / sqrt(1 - cos_b * cos_b);
								Hn -= (cot_a + cot_b) * (P - p1);
							}

							Hn /= (4 * A);
							v->position = v->old_position + data->lambda * Hn;
							v->normal = Hn.normalize();
						}
					}

					spdlog::info("Process success");
				}();
			}

			if (ImGui::Button("Set Normal to Color")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}

					data->mesh->SetToEditable();
					std::vector<rgbf> colors;
					for (auto* v : data->heMesh->Vertices())
						colors.push_back((v->normal.as<valf3>() + valf3{ 1.f }) / 2.f);
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
					std::vector<uint32_t> indices(M * 3);
					for (size_t i = 0; i < N; i++)
						positions[i] = data->heMesh->Vertices().at(i)->position;
					for (size_t i = 0; i < M; i++) {
						auto tri = data->heMesh->Indices(data->heMesh->Polygons().at(i));
						indices[3 * i + 0] = static_cast<uint32_t>(tri[0]);
						indices[3 * i + 1] = static_cast<uint32_t>(tri[1]);
						indices[3 * i + 2] = static_cast<uint32_t>(tri[2]);
					}
					data->mesh->SetPositions(std::move(positions));
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
