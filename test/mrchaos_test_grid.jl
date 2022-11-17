using Test, Agents, Random
using Agents.Graphs, Agents.DataFrames
using Agents.Pathfinding
using StatsBase: mean
using StableRNGs
using PrettyPrint:pprintln

using Distributed
addprocs(2)
@everywhere begin
    using Test, Agents, Random
    using Agents.Graphs, Agents.DataFrames
    using StatsBase: mean
    using StableRNGs
end

mutable struct Agent0 <: AbstractAgent
    id::Int
end

mutable struct Agent1 <: AbstractAgent
    id::Int
    pos::Dims{2}
end

mutable struct Agent2 <: AbstractAgent
    id::Int
    weight::Float64
end

mutable struct Agent3 <: AbstractAgent
    id::Int
    pos::Dims{2}
    weight::Float64
end

mutable struct Agent4 <: AbstractAgent
    id::Int
    pos::Dims{2}
    p::Int
end

mutable struct Agent5 <: AbstractAgent
    id::Int
    pos::Int
    weight::Float64
end

mutable struct Agent6 <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    weight::Float64
end

mutable struct Agent7 <: AbstractAgent
    id::Int
    pos::Int
    f1::Bool
    f2::Int
end

Agent7(id, pos; f1, f2) = Agent7(id, pos, f1, f2)

mutable struct Agent8 <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    f1::Bool
    f2::Int
end

# Agent8(id, pos; f1, f2) = Agent8(id, pos, f1, f2)

# moore = Pathfinding.moore_neighborhood(2)
# vonneumann = Pathfinding.vonneumann_neighborhood(2)
# gspace = GridSpace((5, 5))
# cspace = ContinuousSpace((5., 5.))

# pathfinder = AStar(cspace; walkmap = trues(10, 10))
# model = ABM(Agent6, cspace; properties = (pf = pathfinder,))
# a = add_agent!((0., 0.), model, (0., 0.), 0.)
# @test is_stationary(a, model.pf)

# a.pos = (1.,1.)
# plan_route!(a, (4., 3.), model.pf)

# plan_route!(a, (4., 4.), model.pf)
# @test !is_stationary(a, model.pf)
# @test length(model.pf.agent_paths) == 1
# move_along_route!(a, model, model.pf, 0.35355)
# @test all(isapprox.(a.pos, (4.75, 4.75); atol))

# # test waypoint skipping
# move_agent!(a, (0.25, 0.25), model)
# plan_route!(a, (0.75, 1.25), model.pf)
# move_along_route!(a, model, model.pf, 0.807106)
# @test all(isapprox.(a.pos, (0.75, 0.849999); atol)) || all(isapprox.(a.pos, (0.467156, 0.967156); atol))
# # make sure it doesn't overshoot the end
# move_along_route!(a, model, model.pf, 20.)
# @test all(isapprox.(a.pos, (0.75, 1.25); atol))

# delete!(model.pf.agent_paths, 1)
# @test length(model.pf.agent_paths) == 0

# model.pf.walkmap[:, 3] .= 0
# @test all(get_spatial_property(random_walkable(model, model.pf), model.pf.walkmap, model) for _ in 1:10)
# rpos = [random_walkable((2.5, 0.75), model, model.pf, 2.0) for _ in 1:50]
# @test all(get_spatial_property(x, model.pf.walkmap, model) && euclidean_distance(x, (2.5, 0.75), model) <= 2.0 + atol for x in rpos)

# pcspace = ContinuousSpace((5., 5.); periodic = false)
# pathfinder = AStar(pcspace; walkmap = trues(10, 10))
# model = ABM(Agent6, pcspace; properties = (pf = pathfinder,))
# a = add_agent!((0., 0.), model, (0., 0.), 0.)
# @test all(plan_best_route!(a, [(2.5, 2.5), (4.99,0.), (0., 4.99)], model.pf) .≈ (2.5, 2.5))
# @test length(model.pf.agent_paths) == 1
# move_along_route!(a, model, model.pf, 1.0)
# @test all(isapprox.(a.pos, (0.7071, 0.7071); atol))

# model.pf.walkmap[:, 3] .= 0
# move_agent!(a, (2.5, 2.5), model)
# @test all(plan_best_route!(a, [(3., 0.3), (2.5, 2.5)], model.pf) .≈ (2.5, 2.5))
# @test isnothing(plan_best_route!(a, [(3., 0.3), (1., 0.1)], model.pf))

# kill_agent!(a, model, model.pf)
# @test length(model.pf.agent_paths) == 0

# @test isnothing(penaltymap(model.pf))
# pmap = fill(1, 10, 10)
# pathfinder = AStar(cspace; cost_metric = PenaltyMap(pmap))
# @test penaltymap(pathfinder) == pmap

# ===============
gspace = GridSpace((5,5))
pathfinder = AStar(gspace)
pprintln(pathfinder.cost_metric)
model = ABM(Agent3, gspace; properties = (pf = pathfinder,) )
a = add_agent!((5,2), model, 654.5)
pprintln(a)
pprintln(penaltymap(model.pf))
pmap = fill(1,5,5)
pathfinder = AStar(gspace; cost_metric = PenaltyMap(pmap))
model = ABM(Agent3, gspace;properties = (pf=pathfinder,))
pprintln(model)
@show penaltymap(model.pf)
@show pathfinder.walkmap
display( pathfinder.walkmap)
pathfinder.walkmap[:,3] .= false
@show pathfinder.walkmap
display(pathfinder.walkmap)

npos = collect(nearby_walkable((5, 4), model, model.pf))
ans = [(4, 4), (5, 5), (1, 4), (4, 5), (1, 5)]
@test length(npos) == length(ans)
@test all(x in npos for x in ans)
npos = random_walkable(model, model.pf)
@test all(pathfinder.walkmap[random_walkable(model, model.pf)...] for _ in 1:10)
# ================

# @test all(get_spatial_property(random_walkable(model, model.pf), model.pf.walkmap, model) for _ in 1:10)
# rpos = [random_walkable((2.5, 0.75), model, model.pf, 2.0) for _ in 1:50]
# @test all(get_spatial_property(x, model.pf.walkmap, model) && euclidean_distance(x, (2.5, 0.75), model) <= 2.0 + atol for x in rpos)