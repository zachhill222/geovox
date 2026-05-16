#pragma once


namespace GV
{
	//This function colors a set of active elements given a dofhandler on those elements
	//such that no two elements with the same color are in the support of any single dof
	//This allows most fem assembly loops to run in parallel over elements with the same color
}