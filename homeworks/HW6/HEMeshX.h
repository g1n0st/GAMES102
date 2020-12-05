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
	Ubpa::pointf3 old_position{ 0.f };
	Ubpa::pointf3 position{ 0.f };
	Ubpa::normalf normal{ 0.f };
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
	
	float Area() const {
		auto* p0 = HalfEdge()->Origin();
		auto* p1 = HalfEdge()->Next()->Origin();
		auto* p2 = HalfEdge()->Next()->Next()->Origin();
		auto d01 = p1->old_position - p0->old_position;
		auto d02 = p2->old_position - p0->old_position;
		return 0.5f * d02.cross(d01).norm();
	}
};


struct HEMeshX : Ubpa::HEMesh<HEMeshXTraits> {
	// you can add any attributes and mothods to HEMeshX
};