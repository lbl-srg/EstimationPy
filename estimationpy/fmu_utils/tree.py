'''
Created on Jul 22, 2013

@author: marco
'''

class Tree(object):
    """
    
    Basic implementation of a tree.
    
    This tree is used for representing the components that are part of a Modelica model.
    The structure of the tree is retrieved by the dot notation used in the Modelica model (see method << addFromString(...) >>).
    
    Each tree is an object that can have:
    
    ** a name,
    ** a general object (an other class),
    ** a list of children
    
    If the object has not children it is a leaf of the tree.
    
    The children are of the same class <<Tree>>, thus the methods for adding and retrieving information
    can be done recursively.
    
    """
    
    def __init__(self, name = "", object = None):
        """
        
        Initialization of the tree by providing its name, and a generic object.
        
        """
        self.name = name
        self.object = object
        self.childs = []
    
    def isFather(self):
        """
        
        If the list of children is empty, return False.
        
        """
        return self.childs != []
    
    def getAll(self):
        """
        
        Return a list containing the children.
        
        """
        res = [str(self.name)]
        for child in self.childs:
            res.append(child.getAll())
        return res
    
    def deleteAll(self):
        """
        
        This method deletes all the children in a recursive way (to the lower levels).
        In order to completely delete all the children be sure to run this method from the root of the tree.
        
        """
        for child in self.childs:
            child.deleteAll()
        del self.childs[:]
        
    def getName(self):
        """
        
        Return the name of the considered tree element.
        
        """
        return str(self.name)
    
    def getObject(self):
        """
        
        Return the object associated to the considered tree element.
        
        """
        return self.object
    
    def getChilds(self):
        """
        
        Return the children of this element (not going to deeper levels).
        
        """
        return self.childs
    
    def getChildNames(self):
        """
        
        Return the names of the children of this element (not going to deeper levels).
        
        """
        return [child.getName() for child in self.childs]
    
    def getChild(self, name):
        """
        
        Return the child given the name as search criteria.
        
        """
        for child in self.childs:
            if child.getName( )== name:
                return child
        print "No child with name: "+str(name)+" in father "+str(self.name)
        return None
    
    def addChild(self, name, object = None):
        """
        
        This method adds a child to the list of children.
        The new child will have a name and an associated object (e.g. a generic Class)
        
        """
        if name not in self.getChildNames():
            child = Tree(name, object)
            self.childs.append(child)
        else:
            print "Son <"+str(name)+"> already existing..."
    
    def addFromString(self, names, object = None):
        """
        
        This methods allows to add a child into the tree, given its name specified with the dot notation.
        E.G.
        
        addFromString("one.two.three,four",object1)
        addFromString("one.two.three,five",object2)
        
        will create the following tree structure
        
        one
         |
        two
         |
        three
         |
         |---four, object1
         |---five, object2
         
        A tree with two leaves named "four" and "five" with the object1 and object 2 associated to them.
                
        """
        N = len(names)
        # if the length of the names vector is zero, end
        if N == 0:
            return
        
        # if the length of the names is one, it is the leaf of the tree
        # thus the object should be added
        
        if names[0] not in self.getChildNames():
            if N==1:
                child = Tree(names[0], object)
            else:
                child = Tree(names[0], None)
            
            self.childs.append(child)
            
        self.getChild(names[0]).addFromString(names[1:], object)
                
          

if __name__ == '__main__':
    """
    
    Example for testing the class Tree
    
    """
    
    print "TESTING CLASS: Tree"+"\n"
    
    print "Added root element named: root"
    tree = Tree("root")
    print "Tree: "+str(tree.getAll())+"\n"
    
    print "Added three sons to the root: son1, son2, son3"
    tree.addChild("son1")
    tree.addChild("son2")
    tree.addChild("son3")
    print "Tree: "+str(tree.getAll())+"\n"
    
    print "Added new elements with dot notation"
    dotName1 = "root.son4.grandson1"
    dotName2 = "root.son4.grandson2"
    dotName3 = "root.son5.grandson1"
    print "\t"+dotName1+"\n\t"+dotName2+"\n\t"+dotName3
    
    names1 = dotName1.split(".")
    tree.addFromString(names1)
    names2 = dotName2.split(".")
    tree.addFromString(names2)
    names3 = dotName3.split(".")
    tree.addFromString(names3)
    
    
    print "Tree: "+str(tree.getAll())+"\n"