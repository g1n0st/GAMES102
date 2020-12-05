#pragma once

#include <Utopia/Render/Mesh.h>
#include <Utopia/App/Editor/InspectorRegistry.h>
#include "../HEMeshX.h"

struct DenoiseData {
	[[UInspector::min_value(0.f)]]
	[[UInspector::tooltip("lambda")]]
	float lambda = 0.1f;

	[[UInspector::min_value(0)]]
	[[UInspector::tooltip("iteration time(s)")]]
	int iter_time = 0;

	std::shared_ptr<Ubpa::Utopia::Mesh> mesh;

	[[UInspector::hide]]
	std::shared_ptr<HEMeshX> heMesh{ std::make_shared<HEMeshX>() };

	[[UInspector::hide]]
	Ubpa::Utopia::Mesh copy;
};

#include "details/DenoiseData_AutoRefl.inl"
