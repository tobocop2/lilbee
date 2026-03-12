extends Node

@export var width: int = 30
@export var height: int = 20
@export var wall_tile_source_id: int = 0
@export var floor_tile_source_id: int = 0
@export var wall_atlas_coords: Vector2i = Vector2i(0, 0)
@export var floor_atlas_coords: Vector2i = Vector2i(1, 0)
@export var collectible_scene: PackedScene
@export var collectible_count: int = 10

var tile_map: TileMapLayer
var collectibles_container: Node2D
var astar_grid: AStarGrid2D

enum TileType { FLOOR, WALL }

var grid: Dictionary = {}


func _ready() -> void:
	generate_level()


func generate_level() -> void:
	_setup_astar()
	_generate_maze()
	_place_tiles()
	_place_collectibles()


func _setup_astar() -> void:
	astar_grid = AStarGrid2D.new()
	astar_grid.region = Rect2i(0, 0, width, height)
	astar_grid.cell_shape = AStarGrid2D.CELL_SHAPE_SQUARE
	astar_grid.diagonal_mode = AStarGrid2D.DIAGONAL_MODE_NEVER
	astar_grid.update()


func _generate_maze() -> void:
	for x in range(width):
		for y in range(height):
			grid[Vector2i(x, y)] = TileType.WALL
	
	var maze_generator = _MazeGenerator.new(width, height)
	var floor_tiles = maze_generator.generate()
	
	for tile in floor_tiles:
		grid[tile] = TileType.FLOOR


func _place_tiles() -> void:
	for x in range(width):
		for y in range(height):
			var coords = Vector2i(x, y)
			var tile_type = grid.get(coords, TileType.WALL)
			
			if tile_type == TileType.WALL:
				tile_map.set_cell(coords, wall_tile_source_id, wall_atlas_coords)
			else:
				tile_map.set_cell(coords, floor_tile_source_id, floor_atlas_coords)
				astar_grid.set_point_solid(coords, false)
	
	astar_grid.update()


func _place_collectibles() -> void:
	var floor_cells = []
	for x in range(width):
		for y in range(height):
			var coords = Vector2i(x, y)
			if grid.get(coords, TileType.WALL) == TileType.FLOOR:
				floor_cells.append(coords)
	
	floor_cells.shuffle()
	
	var placed = 0
	for cell in floor_cells:
		if placed >= collectible_count:
			break
		if _is_valid_collectible_position(cell):
			_spawn_collectible(cell)
			placed += 1


func _is_valid_collectible_position(cell: Vector2i) -> bool:
	var neighbors = [
		cell + Vector2i(0, -1),
		cell + Vector2i(0, 1),
		cell + Vector2i(-1, 0),
		cell + Vector2i(1, 0)
	]
	var floor_count = 0
	for neighbor in neighbors:
		if grid.get(neighbor, TileType.WALL) == TileType.FLOOR:
			floor_count += 1
	return floor_count >= 2


func _spawn_collectible(cell: Vector2i) -> void:
	if collectible_scene:
		var collectible = collectible_scene.instantiate()
		collectibles_container.add_child(collectible)
		collectible.position = tile_map.map_to_local(cell)
	else:
		var marker = Marker2D.new()
		collectibles_container.add_child(marker)
		marker.position = tile_map.map_to_local(cell)


func set_tile_map(tile_map_layer: TileMapLayer) -> void:
	tile_map = tile_map_layer


func set_collectibles_container(container: Node2D) -> void:
	collectibles_container = container


func get_random_floor_position() -> Vector2i:
	var floor_cells = []
	for x in range(width):
		for y in range(height):
			var coords = Vector2i(x, y)
			if grid.get(coords, TileType.WALL) == TileType.FLOOR:
				floor_cells.append(coords)
	
	if floor_cells.is_empty():
		return Vector2i(-1, -1)
	
	floor_cells.shuffle()
	return floor_cells[0]


func is_wall(coords: Vector2i) -> bool:
	return grid.get(coords, TileType.WALL) == TileType.WALL


func get_path(from: Vector2i, to: Vector2i) -> PackedVector2Array:
	if is_wall(from) or is_wall(to):
		return PackedVector2Array()
	return astar_grid.get_point_path(from, to)


class _MazeGenerator:
	var width: int
	var height: int
	var rng: RandomNumberGenerator
	var visited: Dictionary = {}
	var floor_tiles: Array[Vector2i] = []

	func _init(w: int, h: int):
		width = w
		height = h
		rng = RandomNumberGenerator.new()
		rng.randomize()

	func generate() -> Array[Vector2i]:
		if width < 5 or height < 5:
			return _generate_simple_room()
		
		_floor_fill(1, 1, width - 2, height - 2)
		
		var start_pos = Vector2i(1, 1)
		_carve_passages_from(start_pos)
		
		_add_random_exits()
		_add_rooms()
		
		return floor_tiles

	func _generate_simple_room() -> Array[Vector2i]:
		var result: Array[Vector2i] = []
		for x in range(1, width - 1):
			for y in range(1, height - 1):
				result.append(Vector2i(x, y))
		return result

	func _floor_fill(x1: int, y1: int, x2: int, y2: int) -> void:
		for x in range(x1, x2 + 1):
			for y in range(y1, y2 + 1):
				var pos = Vector2i(x, y)
				if not pos in visited:
					visited[pos] = true
					floor_tiles.append(pos)

	func _carve_passages_from(start: Vector2i) -> void:
		var stack: Array[Vector2i] = [start]
		visited[start] = true
		
		while not stack.is_empty():
			var current = stack.back()
			var neighbors = _get_unvisited_neighbors(current)
			
			if neighbors.is_empty():
				stack.pop_back()
			else:
				var next = neighbors[rng.randi() % neighbors.size()]
				var mid = (current + next) / 2
				
				if not visited.has(mid):
					visited[mid] = true
					floor_tiles.append(mid)
				
				visited[next] = true
				floor_tiles.append(next)
				stack.append(next)

	func _get_unvisited_neighbors(pos: Vector2i) -> Array[Vector2i]:
		var result: Array[Vector2i] = []
		var directions = [
			Vector2i(0, -2),
			Vector2i(0, 2),
			Vector2i(-2, 0),
			Vector2i(2, 0)
		]
		
		for dir in directions:
			var neighbor = pos + dir
			if _is_valid(neighbor) and not visited.has(neighbor):
				result.append(neighbor)
		
		result.shuffle()
		return result

	func _is_valid(pos: Vector2i) -> bool:
		return pos.x > 0 and pos.x < width - 1 and pos.y > 0 and pos.y < height - 1

	func _add_random_exits() -> void:
		var edge_cells: Array[Vector2i] = []
		
		for x in range(1, width - 1):
			edge_cells.append(Vector2i(x, 0))
			edge_cells.append(Vector2i(x, height - 1))
		
		for y in range(1, height - 1):
			edge_cells.append(Vector2i(0, y))
			edge_cells.append(Vector2i(width - 1, y))
		
		edge_cells.shuffle()
		
		var exit_count = mini(4, edge_cells.size())
		for i in range(exit_count):
			var exit_pos = edge_cells[i]
			if not exit_pos in floor_tiles:
				floor_tiles.append(exit_pos)

	func _add_rooms() -> void:
		var room_count = rng.randi_range(2, 5)
		
		for i in range(room_count):
			var room_w = rng.randi_range(3, 6)
			var room_h = rng.randi_range(3, 6)
			var room_x = rng.randi_range(2, width - room_w - 2)
			var room_y = rng.randi_range(2, height - room_h - 2)
			
			for x in range(room_x, room_x + room_w):
				for y in range(room_y, room_y + room_h):
					var pos = Vector2i(x, y)
					if not pos in floor_tiles:
						floor_tiles.append(pos)
