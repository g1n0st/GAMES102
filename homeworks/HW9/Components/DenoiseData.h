#pragma once

#include <Utopia/Render/Mesh.h>
#include <Utopia/App/Editor/InspectorRegistry.h>
#include "../HEMeshX.h"
#include <set>

struct DenoiseData {
	// [[...]] is attribute list.
	// User-defined attributes are useless for the compiler,
	// but Utopia use these informations for Inspector.
	// If you change struct DenoiseData, you should delete 'details/DenoiseData_AutoRefl.inl'
	// or right click 'DenoiseData_AutoRefl.inl.rule' in 'solution explorer' and select 'compile'
	// CMake will generate a new one.

	// [[UInspector::range(std::pair{0.f, 10.f})]]
	[[UInspector::min_value(0.f)]]
	float scale = 0.65f;

	std::shared_ptr<Ubpa::Utopia::Mesh> mesh;

	[[UInspector::hide]]
	std::shared_ptr<HEMeshX> heMesh{ std::make_shared<HEMeshX>() };

	[[UInspector::hide]]
	Ubpa::Utopia::Mesh copy;

	[[UInspector::hide]]
	std::set<int> facesToRemove;

	[[UInspector::hide]]
	std::set<int> verticesToRemove;

	[[UInspector::hide]]
	std::vector<std::vector<Vertex*> > faces;
};

#include "details/DenoiseData_AutoRefl.inl"
