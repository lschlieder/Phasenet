import numpy as np

def get_numpy_array_from_str(str):
    arr = []
    str_splt = str.split(',')
    for num in str_splt:
        arr.append(float(num))
    np_arr = np.array(arr)
    return np_arr

#####
##Hexagonal coordinate stuff
#####
def cube_to_2d(cube_coord, side_length, orientation='pointy'):
    """
    Convert cube coordinates (q, r, s) to 2D position (x, y) in a hexagonal grid.

    Parameters:
        q (int): The q coordinate in cube coordinates.
        r (int): The r coordinate in cube coordinates.
        s (int): The s coordinate in cube coordinates.
        side_length (float): Length of the hexagon's side.
        orientation (str, optional): Orientation of the hexagons, either 'pointy' or 'flat'. Default is 'pointy'.

    Returns:
        float: The x-coordinate in 2D position.
        float: The y-coordinate in 2D position.
    """
    if orientation not in ['pointy', 'flat']:
        raise ValueError("Invalid orientation. Use 'pointy' or 'flat'.")

    q = cube_coord[0]
    r = cube_coord[1]
    s = cube_coord[2]
    if orientation == 'pointy':
        
        x = np.sqrt(3) * side_length * (q + s / 2)
        y = 3 / 2 * side_length * s
    else:  # flat orientation
        x = 3 / 2 * side_length * q
        y = np.sqrt(3) * side_length * (r + q / 2)

    return np.array([x,y])

def get_ring(cube_coord, radius:int):
    assert(radius > 0)
    directions = np.array([ [1,0,-1], [1,-1,0], [0,-1,1], [-1,0,1], [-1,1,0], [0,1,-1]])
    results = []
    h = cube_coord + directions[4]*radius
    for i in range(0,6):
        for j in range(0,radius):
            results.append(h)
            h = h + directions[i]
    return results

def get_neighbors( cube_coord):
    neighbors = []
    directions = [ [1,0,-1], [1,-1,0], [0,-1,1], [-1,0,1], [-1,1,0], [0,1,-1]]
    for d in directions:
        neighbors.append(cube_coord +d)
    return neighbors

def create_field(n = 10):
    points = []
    starting_point = np.array([0,0,0])
    addable_points = [starting_point]
    
    r = 0 
    while len(points) < n:
        if len(addable_points) > 0:
            points.append(addable_points.pop())
        else:
            r = r+1
            ring = get_ring(starting_point, r)
            addable_points = addable_points + ring
            points.append(addable_points.pop())
    return points  






        

