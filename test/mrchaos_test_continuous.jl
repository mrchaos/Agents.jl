
using Test, Agents, Random
using Agents.Graphs, Agents.DataFrames
using Agents.Pathfinding
using StatsBase: mean
using StableRNGs
using PrettyPrint:pprintln

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

moore = Pathfinding.moore_neighborhood(2)
vonneumann = Pathfinding.vonneumann_neighborhood(2)
cspace = ContinuousSpace((5., 5.))
atol = 0.0001

pathfinder = AStar(cspace;walkmap = trues(10,10))
model = ABM(Agent6, cspace; properties = (pf = pathfinder,))
a = add_agent!((0., 0.), model, (0., 0.), 0.)
plan_route!(a,(4.,4.), model.pf)
move_along_route!(a,model,model.pf,0.35355)
@test all(isapprox.(a.pos,(4.75,4.75);atol))

# test waypoint skipping
move_agent!(a,(0.25,0.25),model)
a.pos
plan_route!(a,(0.75,1.25),model.pf)
collect(model.pf.agent_paths[a.id])
move_along_route!(a,model,model.pf,0.807106)
a.pos
collect(model.pf.agent_paths[a.id])
@show all(isapprox.(a.pos,(0.75,0.849999); atol))
@show all(isapprox.(a.pos,(0.467156, 0.967156); atol))

# make sure it doesn't overshooting the end
move_along_route!(a, model, model.pf, 20.)
a.pos
@show all(isapprox.(a.pos, (0.75, 1.25); atol))
a.id
collect(keys(model.pf.agent_paths))
delete!(model.pf.agent_paths,a.id)
model.pf.agent_paths
@show length(model.pf.agent_paths) == 0

collect(model.pf.walkmap[:,3])
model.pf.walkmap[:,3] .= false
collect(model.pf.walkmap[:,3])
# get_spatial_index
# walkmap에서 3열을 false로 설정 했기 때문에 random_walkable에서 반환되는 좌표는 
# walkmap에서 false로 설정된 영역은 없다.
# 따라서 아래 get_spatial_property의 값은 모두 true로 반환된다.
pos_list = map(x->random_walkable(model,model.pf),1:10)
idxs = map(pos->get_spatial_index(pos,model.pf.walkmap,model),pos_list)
all(model.pf.walkmap[idxs])
all(map(pos->get_spatial_property(pos,model.pf.walkmap,model),pos_list))

pos = (5.,5.)
r = 3.0
random_walkable(pos,model,model.pf)

# non-periodic space
pcspace = ContinuousSpace((5.,5.);periodic=false)
pprintln(pcspace)
pathfinder = AStar(pcspace;walkmap = trues(10,10))
model = ABM(Agent6, pcspace;properties = (pf=pathfinder,))
a = add_agent!((0.,0.), model,(0.,0.),0.)
pprintln(a)
@test all(plan_best_route!(a,[(2.5,2.5),(4.99,0.),(0.,4.99)],model.pf) .≈ (2.5,2.5))
collect(model.pf.agent_paths[a.id])
move_along_route!(a,model,model.pf,1.0)
a.pos
@test all(isapprox.(a.pos,(0.7071,0.7071);atol))

#--------------------------------------------------------------
# agent와 목표 도착지점사이에  긴 장벽이 있는 경우 
# non-periodic space인 경우 목표도착 지점에 
# 도달 할 수 있는 경로는 없다
#----------------------------------------------------------------
# discrete walkmap의 3열에 갈수없는 영역을 표시 즉 긴 장벽을 설정
model.pf.walkmap[:,3] .= false
# walkmap 좌표에서 3열이 continuous space에서 좌표가 어떻게 되는지 계산
# 하기 위해 (1,3)을 넣고 계산하면 (0.25, 1.25)가 나오는데
# 1.25 열에 못가는 영역이 있다
@test all(Pathfinding.to_continuous_position((1,3),model.pf) .≈ (0.25,1.25))
# agent를 2.5,2.5 로 옮긴다.
move_agent!(a,(2.5,2.5),model)
@test all(plan_best_route!(a,[(3.,0.3),(2.5,2.5)],model.pf) .≈ (2.5,2.5))
# 목표 도착지점을 각 각 (3.,0.3),(1.,0.1) 로 잡는 경우
# 목표 도착지점은 장벽이 놓인 1.25 영역 왼쪽에 있다.
pos_list = [(3.,0.3),(1.,0.1)]
# agent의 위치는 (2.5,2.5)로 장벽이 놓인 오른쪽에 있다
# 따라서 현재 agent위치에서 목표 도착지점까지 갈 수 있는 경로는 없다.
@test isnothing(plan_best_route!(a,pos_list,model.pf))

kill_agent!(a,model, model.pf)
@test length(model.pf.agent_paths) == 0

#-----------------------------
# PenaltyMap test
#-----------------------------
# 기본적으로 모델에 penaltymap은 비어 있다.
# penaltymap은 base_metric과 pmap 두개를 가진다.
# 이동 거리 계산시 cost는
# 기본적으로 base_metric이 사용되고 추가적으로 pmap이 사용된다
@test isnothing(penaltymap(model.pf))
# 10x10 penaltymap 생성
pmap = fill(1,10,10)
pathfinder = AStar(cspace; cost_metric = PenaltyMap(pmap))
pprintln(pathfinder)
@test penaltymap(pathfinder) == pmap
# 갈수 없는 영역이 설정되지 않았기 때문에 기본적으로 random_Walkable은
# 전체 영역중 임의의 위치를 반환하는데 해당영역의 속성값은 전부 true
@test all(get_spatial_property(random_walkable(model,model.pf),model.pf.walkmap, model) for _ in 1:10 )
# 중심 (2.5, 0.75),  반지름 2.0 범위안에 있는 임의의 좌표 50개를 추출한다
rpos = [random_walkable((2.5,0.75), model,model.pf,2.0) for _ in 1:50]
# rpos가 전부 갈 수 있는 활성화 영역이고 중심에서 반지름 2.0 안에 있음을 확인함
@test all(get_spatial_property(x,model.pf.walkmap,model) &&
    euclidean_distance(x,(2.5,0.75),model) <= 2.0 + atol for x in rpos)

