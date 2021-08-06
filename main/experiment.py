class experiment:
	#add directories for forget and correct statistics
	def make_directory(dir_name):
		Path(str(parent_dir_path) + "/" + dir_name).mkdir(parents=True, exist_ok=True)