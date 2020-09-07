import os
import os.path
import yaml
import graphviz as gz
from yaml import Loader, Dumper
import re
import sys
from os.path import abspath, join, dirname

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
        elif x[key] is not None:
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


def fill_hole(interfaces):
    """ Add missing attribute to interfaces
    """
    for module_name, interface in interfaces.items():
        for clas in interface.keys():

            check_attr_dict(interfaces[module_name][clas], "requirement")
            check_attr_dict(interfaces[module_name][clas], "public")

            requirement = interfaces[module_name][clas]["requirement"]
            public = interfaces[module_name][clas]["public"]

            check_attr_list(requirement, "module")
            check_attr_list(requirement, "property")
            check_attr_list(requirement, "function")
            check_attr_list(requirement, "property_dependency")

            check_attr_list(public, "property")
            check_attr_list(public, "function")
    

def disambiguate_list(l, spec):
    """ Replace the choice list with a determined choice
    """
    ret = []
    for propert in l:
        if type(propert) is dict:
            prop = list(propert.keys())[0]
            choices = propert[prop]
            if prop not in spec:
                print(prop, " not specified, please choose a semantic meaning")
                exit()
            ret.append(spec[prop])
        else:
            ret.append(propert)
    l = ret
    return ret

def disambiguate(interfaces, specified):
    """
    # Some property can be satisfied by multiple forms, like state_x and cartesian_x,
                # The requirement is met as long as one of them is provided.
                # So we allow the requirement to be the format like {x:[state_x, cartesian_x]}
                # This dict has only one key and one value (list of choices).
                # The user should specify the choice.
    """
    for module_name, interface in interfaces.items():
        if module_name not in specified.keys():
            continue
        for clas in interface.keys():
            if clas not in specified[module_name].keys():
                continue

            requirement = interfaces[module_name][clas]["requirement"]
            public = interfaces[module_name][clas]["public"]

            requirement["property"] = disambiguate_list(requirement["property"], specified[module_name][clas])
            requirement["function"] = disambiguate_list(requirement["function"], specified[module_name][clas])
            public["property"] = disambiguate_list(public["property"], specified[module_name][clas])
            public["function"] = disambiguate_list(public["function"], specified[module_name][clas])
            
def parse(specified):
    """ Parse all inteface.yml and build the connections based on specified
    """

    # find all interface.yml under all modules
    interfaces = {}
    module_root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..') #TODO: change this after reorganization
    for dirpath, dirnames, filenames in os.walk(module_root_path):
        inteface_yamls = [f for f in filenames if f.endswith("interface.yml")]
        for filename in inteface_yamls:
            path = os.path.join(dirpath, filename)
            f = open(path, 'r')
            data = yaml.load(f, Loader=Loader)
            interfaces[data["module"]] = data["class"]

    # The interface supports class inheritance. Get the inheritance for classes first.
    inheritance(interfaces)

    # specified = ["ControllerTest", "PlannerTest", "ModelTest"]
    
    fill_hole(interfaces)

    # disambiguate multiple choice properties
    disambiguate(interfaces, specified)

    module_reqs = {}
    edges = {}
    groups = {}
    given_modules = set()
    property_deps = {}
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
           
            requirement = interface[clas]["requirement"]
            public = interface[clas]["public"]

            for module_req in requirement["module"]:
                if module_req not in specified.keys():
                    print("Missing specification: ",module_req)
                    exit()

                module_reqs[module_name].append(module_req)

            for propert in requirement["property"] + requirement["function"]:
                add_edge(edges, propert, module_name)

            for propert in public["property"] + public["function"]:
                add_edge(edges, module_name, propert)
            
            for deps in requirement["property_dependency"]:
                prop = list(deps.keys())[0]
                for dep in deps[prop]:
                    add_edge(property_deps, dep, prop)

        if not used:
            module_reqs.pop(module_name)

    return edges, module_reqs, groups, property_deps

def get_nodes_modules_properties_in_edges(edges, module_reqs):
    """Get some useful variables
    """
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
        if a not in modules:  # a is not module, then bs are all modules
            properties.add(a)
            continue  
        # a is module, then bs are all properties
        for b in bs:
            nodes.add(b)
            check_attr_list(in_edges, b)
            in_edges[b].append(a)
            if b not in modules:
                properties.add(b)
    
    return nodes, modules, properties, in_edges


def sort_module(module_reqs):
    """Sort modules based on dependency relations.
    """
    module_sorted = []
    module_depth = {}
    def dfs(module_name):
        max_depth = 0
        if module_name in module_reqs.keys():
            for req in module_reqs[module_name]:
                max_depth = max(max_depth, dfs(req)+1)
        module_depth[module_name] = max_depth
        return max_depth

    for req in module_reqs:
        dfs(req)
    
    for depth in range(len(module_reqs.keys())):
        for module, mdepth in module_depth.items():
            if mdepth == depth:
                module_sorted.append(module)
    return module_sorted


def draw_deps(edges, module_reqs, groups, property_deps):
    g = gz.Digraph('G', filename=join(abspath(dirname(__file__)), '../documentation/img/deps.gv'))
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
        g.node(node, shape='box', style="filled", fillcolor="peachpuff", color="peachpuff")

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
            g.edge(a, b, color="gray50")

    # draw dependencies between properties
    for a in property_deps.keys():
        for b in property_deps[a]:
            if a in properties or b in properties:
                g.edge(a, b, color="gray80", weight="100")
    

    # draw invisible dependencies of modules, to draw hiearchy structure
    module_list = sort_module(module_reqs)
    for i in range(len(module_list)-1):
        g.edge(module_list[i], module_list[i+1], style='invis', weight='1000')
    
    for i in range(len(module_list)-1):
        for output in edges[module_list[i]]:
            g.edge(output, module_list[i+1], style='invis')
    

    # make outputs of a module show in the same level
    for module in module_list:
        with g.subgraph() as s:
            s.attr(rank='same')
            for b in edges[module]:
                s.node(b)

    g.view()
    
def show_architecture(specified):
    edges, module_reqs, groups, property_deps = parse(specified)
    draw_deps(edges, module_reqs, groups, property_deps)