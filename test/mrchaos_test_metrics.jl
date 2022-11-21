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
gspace = GridSpace((5, 5))
cspace = ContinuousSpace((5., 5.))
atol = 0.0001

# struct AStar{D,P,M,T,C<:CostMetric{D}} <: GridPathfinder{D,P,M}
# function AStar{D,P,M,T,C}(
#    agent_paths::Dict,
#    dims::NTuple{D,T},
#    neighborhood::Vector{CartesianIndex{D}},
#    admissibility::Float64,
#    walkmap::BitArray{D},
#    cost_metric::C,
# ) where {D,P,M,C,T}
#
# D : dimension, P : periodic, M : diagonal_movement, T : type
# C :  cost metric type

pfinder_2d_np_m = AStar{2,false,true,Int64,DirectDistance{2}}(
    Dict(),
    (10,10),
    copy(moore),
    0.0,
    trues(10,10),
    DirectDistance{2}(),
)
pfinder_2d_np_nm = AStar{2,false,false,Int64,DirectDistance{2}}(
    Dict(),
    (10,10),
    copy(vonneumann),
    0.0,
    trues(10,10),
    DirectDistance{2}(),
)
pfinder_2d_p_m = AStar{2,true,true,Int64,DirectDistance{2}}(
    Dict(),
    (10,10),
    copy(moore),
    0.0,
    trues(10,10),
    DirectDistance{2}(),
)
pfinder_2d_p_nm = AStar{2,true,false,Int64,DirectDistance{2}}(
    Dict(),
    (10,10),
    copy(vonneumann),
    0.0,
    trues(10,10),
    DirectDistance{2}(),
)

#-------------------------
# 경로의 cost 계산 (DirectDistance)
#-------------------------
# DirectDistance의 cost는 대각선방향은 14, 직각방향은 10 으로 정의 되어 있다.
# (1,1) -14-> (2,2) -14-> (3,3) -14-> (4,4) -10-> (4,5) -10-> (4,6) : 62
@test delta_cost(pfinder_2d_np_m,pfinder_2d_np_m.cost_metric,(1,1),(4,6)) == 62
@test delta_cost(pfinder_2d_np_m,(1,1),(4,6)) == 62
@test delta_cost(pfinder_2d_np_m,DirectDistance{2}(),(1,1),(4,6)) == 62
# periodic이기 때문에
# (1,1) -14-> (10,10) -14-> (9,9) -14-> (8,8) -10-> (8,7) -10-> (8,6)
@test delta_cost(pfinder_2d_p_m,(1,1),(8,6)) == 62

# 직각방향으로만 움직이는 경우
# cost=80 
# (1,1)-10->(2,1)-10->(3,1)-10->(4,1)-10->(4,2)-10->(4,3)-10->(4,4)-10->(4,5)-10->(4,6)
@test delta_cost(pfinder_2d_np_nm,(1,1),(4,6)) == 80
#cost=80 : (1,1)-10->(10,1)-10->(9,1)-10->(8,1)-10->(8,2)-10->(8,3)
#          -10->(8,4)-10->(8,5)-10->(8,6)
@test delta_cost(pfinder_2d_p_nm,(1,1),(8,6)) == 80

#-------------------------
# 경로의 cost 계산 (MaxDistance)
# delta_cost = max(position_dalta)
# position_delta중 가장큰 값을 비용으로 사용한다.
#-------------------------
pprintln(MaxDistance{2}())
# (1,1) -> (2,2) -> (3,3) -> (4,4) -> (4,5) -> (4,6)
# position_dalta = (3,5)
@test delta_cost(pfinder_2d_np_m,MaxDistance{2}(),(1,1), (4,6)) == 5
# (1,1) -> (10,10) -> (9,9) -> (8,8) -> (8,7) -> (8,6)
# position_dalta = (3,5)
@test delta_cost(pfinder_2d_p_m,MaxDistance{2}(),(1,1), (8,6)) == 5
# (1,1) -> (2,1) -> (3,1) -> (4,1) -> (4,2) -> (4,3) -> (4,4) -> (4,5) -> (4,6)
# position_dalta = (3,5)
@test delta_cost(pfinder_2d_np_nm,MaxDistance{2}(),(1,1),(4,6)) == 5
# (1,1)->(10,1)->(9,1)->(8,1)->(8,2)->(8,3)->(8,4)->(8,5)->(8,6)
# position_dalta = (3,5)
@test delta_cost(pfinder_2d_p_nm,MaxDistance{2}(),(1,1),(8,6)) == 5

#-------------------------
# 경로의 cost 계산 (PenaltyMap)
# penaltymap의 base cost metric으로 먼저 cost를 계산하고
# 그기에 더해 pmap에서 from , to 까지 차이의 절대값을 cost로 
# 추가적 더한다.
# delta_cost(pathfinder, metric.base_metric, from, to) +
# abs(metric.pmap[from...] - metric.pmap[to...])
#-------------------------
pmap = fill(0,10,10)
pmap[:,6] .= 100
pmap[1,6] = 0

# delta_cost(pfinder_2d_np_m,DirectDistance{2}(), (1,1), (4,6)) = 62
# (1,1)-14->(2,2)-14->(3,3)-14->(4,4)-10->(4,5)-10->(4,6) : 62
# abs(pmap[1,1] - pmap[4,6]) = abs(0-100) = 100
# PenaltyMap delta_cost = 162
@test delta_cost(pfinder_2d_np_m,PenaltyMap(pmap),(1,1),(4,6)) == 162
# delta_cost(pfinder_2d_p_m,DirectDistance{2}(), (1,1), (8,6)) = 62
# (1,1)-14->(10,10)-14->(9,9)-14->(8,8)-10->(8,7)-10->(8,6) : 62
# abs(pmap[1,1] - pmap[8,6]) = abs(0-100) = 100
# PenaltyMap delta_cost = 162
@test delta_cost(pfinder_2d_p_m,PenaltyMap(pmap),(1,1),(8,6)) == 162
# delta_cost(pfinder_2d_np_nm,DirectDistance{2}(), (1,1), (4,6)) = 80
# (1,1)-10->(2,1)-10->(3,1)-10->(4,1)-10->(4,2)-10->(4,3)-10->(4,4)-10->(4,5)-10->(4,6)
# abs(pmap[1,1] - pmap[4,6]) = abs(0-100) = 100
# PenaltyMap delta_cost = 180
@test delta_cost(pfinder_2d_np_nm,PenaltyMap(pmap),(1,1),(4,6)) == 180
# delta_cost(pfinder_2d_p_nm,DirectDistance{2}(), (1,1), (8,6)) = 62
# (1,1)-10->(10,1)-10->(9,1)-10->(8,1)-10->(8,2)-10->(8,3)-10->(8,4)-10->(8,5)-10->(8,6)
# abs(pmap[1,1] - pmap[8,6]) = abs(0-100) = 100
# PenaltyMap delta_cost = 180
@test delta_cost(pfinder_2d_p_nm,PenaltyMap(pmap),(1,1),(8,6)) == 180


