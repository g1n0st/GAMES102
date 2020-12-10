#pragma once

#include <UHEMesh/HEMesh.h>

#include <UGM/UGM.h>

struct Vertex;
struct Edge;
struct Triangle;
struct HalfEdge;

using HEMeshXTraits = Ubpa::HEMeshTraits<Vertex, Edge, Triangle, HalfEdge>;

struct Vertex : Ubpa::TVertex<HEMeshXTraits> {
	// you can add any attributes and mothods to Vertex
	int idx{ -1 };
	int bidx{ -1 }; // idx on boundary, guarantee continuous on the boundary
	Ubpa::pointf3 position{ 0.f };
	Ubpa::normalf normal{ 0.f };
	Ubpa::pointf2 uv{ 0.f };
};

struct Edge : Ubpa::TEdge<HEMeshXTraits> {
};

struct HalfEdge : Ubpa::THalfEdge<HEMeshXTraits> {
	// you can add any attributes and mothods to HalfEdge

};

struct Triangle : Ubpa::TPolygon<HEMeshXTraits> {
	bool IsTriangle() const {
		return Degree() == 3;
	}
};


struct HEMeshX : Ubpa::HEMesh<HEMeshXTraits> {
	// you can add any attributes and mothods to HEMeshX
};