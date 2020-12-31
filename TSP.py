import math
import random
import time
from collections import defaultdict
from PriorityQueue import PriorityQueue
import numpy as np

class Graph:

    def __init__(self,vertices):
        self.nodes = [] 
        length = len(vertices)
        for i in range(length):
            self.nodes.append(vertices[i])
        self.edges = [] # lines connecting nodes
        self.searchType = 1 #1 for A*, 2 for uniform, 3 for best first, 4 for using my heuristic with A*
        self.successors = defaultdict(list) #dictionary containing destination and weight for each successor

    def addEdge(self,u,v,w,t): 
        for edges in self.edges:
            if (u==edges[0] and v==edges[1]): #checking if there is already an edge connecting the two cities
                print("Edge added before")
                return
            
        self.edges.append([u,v,w,t])
        self.successors[u].append((v, w))   #adding the city v as successor to city u
    
    def searchPath(self, initial_node):
        if not self.edges:
            print('Error: No edges found!')
            return

        print("Initial node:",initial_node)
        if initial_node not in self.nodes:
            print('Error: No nodes found!')
            return

        #getting function cost of the root
        queue = PriorityQueue()
        g_cost = 0
        h_cost = self.PrimsMST(initial_node)
        f_cost = g_cost + h_cost
        queue.insert((initial_node, g_cost, h_cost), f_cost)
        
        visited_nodes = []  #represents our path
        visited_nodes.append(initial_node)
        while len(visited_nodes) < len(self.nodes):
            print("==============================================")
            if queue.isEmpty():
                return 0,[]
            
            current_node, g_cost, h_cost = queue.remove()
            # getting and filtering the successors of the current node
            successors = self.successors[current_node]  
            remaining_successors=[]
            for successor in successors:
                if successor[0] not in visited_nodes:
                    remaining_successors.append(successor)
            print("Child of",current_node,": ",remaining_successors)
            
            remaining_nodes = []    #unvisited nodes
            for i in self.nodes:
                if i not in visited_nodes:
                    remaining_nodes.append(i)

            minimum = math.inf
            a,b,c,d=0,0,0,0
            if len(remaining_successors)!= 0:
                for successor in remaining_successors:
                    destination, weight = successor # dictionary containing destination and weight
                    new_g_cost = 0 if (self.searchType == 3) else (g_cost + weight)
                    h_cost = 0 if (self.searchType == 2) else (self.getHeuristicCost(remaining_nodes,destination))
                    f_cost = new_g_cost + h_cost
                    print("Child:",destination,"f_cost:",f_cost)
                    if f_cost < minimum :
                        minimum = f_cost
                        a,b,c,d=destination,f_cost,new_g_cost,h_cost

                print("Chosen child:",a,"f_cost:",b)
                destination=a
                f_cost=b
                new_g_cost=c
                h_cost=d
                visited_nodes.append(destination)   #adding to path
                queue.insert((destination, new_g_cost, h_cost), f_cost) #adding to queue to expand later
                print("Visited nodes till now:",visited_nodes)

        if len(visited_nodes) == len(self.nodes):
            return visited_nodes
        

    def getHeuristicCost(self,remaining_nodes,destination):
        h_cost = 0
        if len(remaining_nodes)!=1:
            # create a temorary graph with the remaining nodes and their edges 
            temp_graph = Graph(remaining_nodes)
            for temp_edge in self.edges:
                if temp_edge[0] in remaining_nodes and temp_edge[1] in remaining_nodes:
                    temp_graph.addEdge(temp_edge[0],temp_edge[1],temp_edge[2],temp_edge[3]) 
            h_cost = temp_graph.myTrafficHeuristic(destination) if (self.searchType == 4) else temp_graph.PrimsMST(destination) #getting the heuristic function of the successor
        return h_cost
    
    def get_total_distance(self,visited_nodes):
        if len(visited_nodes)<=1:
            return 0
        else:
            total_distance=0
            i=1
            while i<len(visited_nodes):
                for temp_edge in self.edges:
                    if visited_nodes[i-1]==temp_edge[0] and visited_nodes[i]==temp_edge[1]:
                        total_distance=total_distance+temp_edge[2]
                i=i+1
            for temp_edge in self.edges:
                    if visited_nodes[0]==temp_edge[0] and visited_nodes[len(visited_nodes) - 1]==temp_edge[1]:
                        total_distance=total_distance+temp_edge[2]
            return total_distance

    def get_total_time(self,visited_nodes):
        if len(visited_nodes)<=1:
            return 0
        else:
            total_time=0
            i=1
            while i<len(visited_nodes):
                for temp_edge in self.edges:
                    if visited_nodes[i-1]==temp_edge[0] and visited_nodes[i]==temp_edge[1]:
                        total_time=total_time+temp_edge[3]
                i=i+1
            for temp_edge in self.edges:
                    if visited_nodes[0]==temp_edge[0] and visited_nodes[len(visited_nodes) - 1]==temp_edge[1]:
                        total_time=total_time+temp_edge[3]
            return total_time

    def myTrafficHeuristic(self,source):
        if len(self.successors)==0:
            if self.nodes != 0:
                return 1000 #returns big cost if there is dead end
            return 0

        min = math.inf
        destination = -1
        weight = 0
        for edge in self.edges:
            if(edge[0] == source):
                h = edge[2] + edge[3]
                if h < min:
                    min = h
                    destination = edge[1]
                    weight = h

        remaining_nodes = []
        for node in self.nodes:
            if node != source:
                remaining_nodes.append(node)
        temp_graph = Graph(remaining_nodes)
        for temp_edge in self.edges:
            if temp_edge[0] in remaining_nodes and temp_edge[1] in remaining_nodes:
                temp_graph.addEdge(temp_edge[0],temp_edge[1],temp_edge[2],temp_edge[3]) 
        return weight + temp_graph.myTrafficHeuristic(destination)

    def PrimsMST(self,source):
        priority_queue = { source : 0 }
        added = [False] * len(self.edges)
        min_span_tree_cost = 0

        while priority_queue :
            # Choose the adjacent node with the least edge cost
            node = min(priority_queue, key=priority_queue.get)
            cost = priority_queue[node]

            del priority_queue[node]
            if node < len(added) and added[node] == False :
                min_span_tree_cost += cost
                added[node] = True
                for item in self.successors[node] :
                    adjnode = item[0]
                    adjcost = item[1]
                    if adjnode < len(added) and added[adjnode] == False :
                        priority_queue[adjnode] = adjcost

        return min_span_tree_cost

def problemGenerator(n):
    vertices = []
    nodes = []
    for i in range(n):
        x = random.randint(0,50)
        y = random.randint(0,50)
        vertices.append([i,x,y])
        nodes.append(i)
    
    print('Nodes with coordinates: ',vertices)
    g=Graph(nodes)
    for i in range(n):
        for j in range(i+1,n):
            traffic = random.randint(1,60)    #to generate traffic of the road needed for my heuristic
            distance = round(math.sqrt(((vertices[i][1] - vertices[j][1]) ** 2) + ((vertices[i][2] - vertices[j][2]) ** 2)))
            g.addEdge(i,j,distance,traffic)
            g.addEdge(j,i,distance,traffic)
    

    for search in range(1,5):
        start_time = time.time()
        g.searchType = search
        printType = ""
        if search == 1:
            printType = 'A* Search'
        elif search ==2:
            printType = 'Uniform Search'
        elif search ==3:
            printType = 'Best First Search'
        else:
            printType = 'Traffic Heuristic with A* Search'

        print('===============',printType,'=================')
        route = g.searchPath(0) # executes the algorithm
        total_distance = g.get_total_distance(route)
        total_time = g.get_total_time(route)
        if total_distance:
            print("========================================")
            print('Route found for ',printType,':')
            for i in range(len(route)-1):
                print(route[i],"->",route[i+1])
            print(route[len(route)-1],"->",route[0])
            print("total_distance: ",total_distance)
            print("total_time: ",total_time)
            print("total_cost: ",total_distance + total_time)
        else:
            print('Did not find a route')
        print('Execution time for performance: ',time.time() - start_time)
    return

n = input("Enter number of nodes: ")
problemGenerator(int(n))