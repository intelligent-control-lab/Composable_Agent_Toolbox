import networkx as nx
import matplotlib.pyplot as plt
from grave import plot_network
from grave.style import use_attributes
import os
import os.path
import yaml
import graphviz as gz
from yaml import Loader, Dumper
import re


def add_edge(edges, a, b):
    if a not in edges.keys():
        edges[a] = []
    edges[a].append(b)

def check_attr_list(dic, attr):
    dic[attr] = [] if attr not in dic.keys() or dic[attr] is None else dic[attr]

def check_attr_dict(dic, attr):
    dic[attr] = {} if attr not in dic.keys() or dic[attr] is None else dic[attr]


def inheritance(interfaces):
    """ Execute inheritance

    Suppose an entry named "Sbsb(Baba)" is in the interface.
    We will create a new entry named "Sbsb" with all additional requirements 
    inheritated from "Baba" entry.
    """
    for module_name, interface in interfaces.items():
        for clas in list(interface.keys()):
            words = re.split('\W+',clas)
            if len(words) == 2 or words[0] in interface.keys(): # no base class or already merged
                continue
            merge(interface, words[0], words[1]) 

def merge_dict_list(merged, x):
    """ merge x into merged recursively.

    x is either a dict or a list
    """
    if type(x) is list:
        return merged + x

    for key in x.keys():
        if key not in merged.keys():
            merged[key] = x[key]
        else:
            merged[key] = merge_dict_list(merged[key], x[key])
            
    return merged

def merge(interface, subc, base):
    """ merge the base class attrs into subclass attrs.

    If the base class inheritate from other class, get base class merged first.
    """
    words = re.split('\W+',base)
    if len(words) == 3:
        merge(interface, base, words[1])

    merged = interface[subc+"("+base+")"].copy()
    merge_dict_list(merged, interface[base])
    interface[subc] = merged    

def parse():
    interfaces = {}
    for dirpath, dirnames, filenames in os.walk("."):
        inteface_yamls = [f for f in filenames if f.endswith("interface.yml")]
        for filename in inteface_yamls:
            path = os.path.join(dirpath, filename)
            f = open(path, 'r')
            data = yaml.load(f, Loader=Loader)
            interfaces[data["module"]] = data["class"]

    # the interfaces are written in inheritance manner. Get the inheritance of classes first.
    inheritance(interfaces)

    spec = {
        "model":     {"type":"LinearModel",     "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.01, "disc_flag":1}},
        "estimator": {"type":"NaiveEstimator",  "spec":{}},
        "planner":   {"type":"OptimizationBasedPlanner",    "spec":{"horizon":10, "replanning_cycle":10, "dim":2, "n_ob":0}},
        "controller":{"type":"Controller",      "spec":{"feedback": "PID", "params": { "feedback": { "kp": [1, 1], "ki": [0, 0], "kd": [0.5, 0.5] } }}},
    }
    
    specified = {
        "controller":{"PID":{"x":"state_x", "goal_x":"state_goal_x"}, },
        "planner":{"OptimizationBasedPlanner":{{"x":"cartesian_x", "goal_x":"cartesian_goal_x"}}}, 
        "model":{"ModelBase":{}}
    }
    # specified = ["ControllerTest", "PlannerTest", "ModelTest"]
    
    module_reqs = {}
    edges = {}
    groups = {}
    given_modules = set()
    property_deps = {}
    #TODO: ask users to disambiguate properties with multiple choices.

    # go through yamls of user specified modules
    for module_name, interface in interfaces.items():
        if module_name not in specified.keys():
            continue
        given_modules.add(module_name)
        module_reqs[module_name] = []
        used = False
        for clas in interface.keys():
            if clas not in specified[module_name].keys():
                continue
            used = True
            # print("====after continue")
            
            check_attr_dict(interface[clas], "requirement")
            check_attr_dict(interface[clas], "public")

            requirement = interface[clas]["requirement"]
            public = interface[clas]["public"]

            check_attr_list(requirement, "module")
            check_attr_list(requirement, "property")
            check_attr_list(requirement, "function")
            check_attr_list(requirement, "property_dependency")

            check_attr_list(public, "property")
            check_attr_list(public, "function")

            for module_req in requirement["module"]:
                module_reqs[module_name].append(module_req)
            
            for propert in requirement["property"] + requirement["function"]:
                # Some property can be satisfied by multiple forms, like state_x and cartesian_x,
                # The requirement is met as long as one of them is provided.
                # So we allow the requirement to be the format like {x:[state_x, cartesian_x]}
                # This dict has only one key and one value (list of choices).
                if type(propert) is dict:
                    prop = list(propert.keys())[0]
                    choices = propert[prop]
                    groups[prop] = choices
                    add_edge(edges, prop, module_name)
                else:
                    add_edge(edges, propert, module_name)
            
            for propert in public["property"] + public["function"]:
                # A module can provide multiple forms of a state in the same time.
                if type(propert) is dict:
                    prop = list(propert.keys())[0]
                    choices = propert[prop]
                    for choice in choices:
                        add_edge(edges, module_name, choice)
                else:
                    add_edge(edges, module_name, propert)
            
            for deps in requirement["property_dependency"]:
                prop = list(deps.keys())[0]
                for dep in deps[prop]:
                    add_edge(property_deps, dep, prop)

        if not used:
            module_reqs.pop(module_name)

    # print(edges)
    return edges, module_reqs, groups, property_deps
            
#     def set_rank(module):
#         if module in drawed:
#             return
#         drawed.append(module)
#         for req_module in module_reqs[module]:
#             set_rank(req_module)
        
#         print("===module")
#         with g.subgraph() as s:
#             s.attr(rank='same')
#             s.node(module)
#             print(module)
        
#         print("===output")
#         check_attr_list(edges, module)
#         if len(edges[module]) == 0:
#             return
#         with g.subgraph() as s:
#             s.attr(rank='same')
#             check_attr_list(edges, module)
#             for b in edges[module]:
#                 # if b in belongs:
#                 #     b = belongs[b]
#                 s.node(b)
#                 print(b)   
#    # for module in module_reqs.keys():
#    #     set_rank(module)

def get_nodes_modules_properties_in_edges(edges, module_reqs):

    nodes = set()
    properties = set()
    in_edges = {}
    modules = set()
    for a in module_reqs.keys():
        modules.add(a)
        for b in module_reqs[a]:
            modules.add(b)

    for a,bs in edges.items():
        nodes.add(a)
        if a not in modules:
            properties.add(a)
        for b in bs:
            nodes.add(b)
            check_attr_list(in_edges, b)
            in_edges[b].append(a)
            if b not in modules:
                properties.add(a)
    
    return nodes, modules, properties, in_edges



def draw_deps(edges, module_reqs, groups, property_deps):
    g = gz.Digraph('G', filename='img/deps.gv')
    g.attr(compound='true', rankdir="LR")

    # group multiple choice properties into one node
    belongs = {}
    for group in groups.keys():
        with g.subgraph(name='cluster_'+group) as s:
            s.node(group, style="invis", shape="point") # invisible dummy node
            for node in groups[group]:
                s.node(node, shape="box")
                belongs[node] = group
            s.attr(label=group)


    nodes, modules, properties, in_edges = get_nodes_modules_properties_in_edges(edges, module_reqs)
    has_source = in_edges.keys()
    
    # set module nodes default style
    for node in modules:
        g.node(node, shape='box', style="filled", fillcolor="lightskyblue", color="lightskyblue")

    # set property nodes default style
    for node in properties:
        if node not in groups.keys(): # group node is virtual node
            g.node(node, shape='box')

    # find unmet requirements
    for node in properties:
        if node not in has_source and node not in groups.keys() and node not in modules:  # group node is virtual node
            g.node(node, shape='box', style="filled", fillcolor="pink", color="pink")

    # find unmet requirements between properties
    for a in property_deps.keys():
        for b in property_deps[a]:
            if a in has_source and b not in has_source:
                g.node(a, shape='box', style="filled", fillcolor="pink", color="pink")

    # draw dependencies
    for a,bs in edges.items():
        for b in bs:
            if b in nodes:
                nodes.remove(b)
                if b in belongs.keys() and belongs[b] in nodes:
                    nodes.remove(belongs[b])
            
            if a in module_reqs.keys(): # a is a module, then b is a property
                g.node(b, shape='box', style="filled", fillcolor="peachpuff", color="peachpuff") # a is a module ,b is an output
            
            if a in groups.keys(): # a is a property having multiple choices
                g.edge(a, b, ltail = 'cluster_'+a, color='gray50')
            else:
                g.edge(a, b, color="gray50")

    # draw invisible dependencies of modules, to draw hiearchy structure
    for module  in module_reqs.keys():
        for req_module in module_reqs[module]:
            g.edge(req_module, module, style='invis')
    

    g.view()
    

if __name__ == "__main__":
    edges, module_reqs, groups, property_deps = parse()
    draw_deps(edges, module_reqs, groups, property_deps)
