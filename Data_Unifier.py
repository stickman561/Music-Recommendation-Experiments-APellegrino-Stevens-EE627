# This program will unite the data from all of our separate files into one.

import polars as pl

# Create number parsers that can handle the "None" string in the files
def parse_integer_value(value):
    return None if value is None or value == 'None' else int(value)

def parse_float_value(value):
    return None if value is None or value == 'None' else float(value)

# Album data parser
def load_album_data(file_path):
    # Need to be cautious here - we're loading the entire file at once, could block program if too large
    
    with open(file_path, 'r') as album_file:
        lines = album_file.readlines()
    
    # Finding the maximum number of genres in the dataset
    max_genres = max(len(line.strip().split('|')[2:]) for line in lines if '|' in line.strip())
    
    # Fixed columns
    album_ids = []
    artist_ids = []
    genre_lists = []
    
    # Read file line-by-line
    for line in lines:
        
        # Read known columns
        parts = line.strip().split('|')
        album_ids.append(parse_integer_value(parts[0]))
        artist_ids.append(parse_float_value(parts[1]))
        
        # Parse genre parts as floats, and pad with None if there are fewer genres than max_genres
        genre_parts = [parse_float_value(g) for g in parts[2:]]
        genre_parts.extend([None] * (max_genres - len(genre_parts)))
        genre_lists.append(genre_parts)
    
    # Organize data to place into DataFrame
    data = {
        "AlbumID": album_ids,
        "ArtistID": artist_ids
    }
    
    # Variable column width (genres)
    for i in range(max_genres):
        data[f"Genre_{i+1}"] = [genre[i] for genre in genre_lists]
    
    # Define the schema for the DataFrame
    album_schema = {
        "AlbumID": pl.Int64,
        "ArtistID": pl.Float64,
        **{f"Genre_{i+1}": pl.Float64 for i in range(max_genres)}
    }
    
    # Create the DataFrame with the schema
    df = pl.DataFrame(data, schema=album_schema)
    return df

# Artist data parser
def load_artist_data(file_path):
    # Artist data is just a list - no splitting or variable columns needed.
    artist_ids = []
    
    with open(file_path, 'r') as artist_file:
        for line in artist_file:
            artist_ids.append(int(line.strip()))
    
    # Place into DataFrame for easier interop later
    df = pl.DataFrame({
        'ArtistID': artist_ids
    })
    return df

# Genre data parser
def load_genre_data(file_path):
    # List, similar to artist data.
    genre_ids = []
    with open(file_path, 'r') as genre_file:
        for line in genre_file:
            genre_ids.append(int(line.strip()))
    
    # Once again, placed into DataFrame for easier interop
    df = pl.DataFrame({
        'GenreID': genre_ids
    })
    return df

# Test data parser
def load_test_data(file_path):
    
    # Again, reading entire file at once due to variable columns. Must
    # be careful not to exceed system memory or lock CPU thread.
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Fixed columns
    user_ids = []
    track_ids = []
    
    # Handle the unique User|Count format of this file (also seen in training data)
    i = 0
    while i < len(lines):
        user_id, n = map(int, lines[i].strip().split('|'))
        i += 1 
        for _ in range(int(n)):
            track_id = int(lines[i].strip())
            user_ids.append(user_id)
            track_ids.append(track_id)
            i += 1
    
    # DataFrame for interop
    df = pl.DataFrame({
        'UserID': user_ids,
        'TrackID': track_ids
    })
    return df

# Track data parser
def load_track_data(file_path):
    
    # Memory warning; see prior functions
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Finding the maximum number of genres in the dataset
    max_genres = max(len(line.strip().split('|')[3:]) for line in lines if '|' in line.strip())
    
    # Fixed columns
    track_ids = []
    album_ids = []
    artist_ids = []
    genre_lists = []
    
    for line in lines:
        # Read known columns
        parts = line.strip().split('|')
        track_ids.append(parse_integer_value(parts[0]))
        album_ids.append(parse_float_value(parts[1]))
        artist_ids.append(parse_float_value(parts[2]))
        
        # Parse genre parts as floats, and pad with None if there are fewer genres than max_genres
        genre_parts = [parse_float_value(g) for g in parts[3:]]
        genre_parts.extend([None] * (max_genres - len(genre_parts)))
        genre_lists.append(genre_parts)
    
    # Organize data to place into DataFrame
    data = {
        'TrackID': track_ids,
        'AlbumID': album_ids,
        'ArtistID': artist_ids,
    }
    for i in range(max_genres):
        data[f"Genre_{i+1}"] = [genre[i] for genre in genre_lists]
    
    # Define DataFrame schema
    track_data_schema = {
        'TrackID': pl.Int64,
        'AlbumID': pl.Float64,
        'ArtistID': pl.Float64,
        **{f'Genre_{i+1}': pl.Float64 for i in range(max_genres)}  # Fixed dictionary comprehension
    }
    
    # Create final DataFrame
    df = pl.DataFrame(data, schema=track_data_schema)
    
    return df

# Training data parser
def load_training_data(file_path, artist_data, genre_data, album_data):
    
    # Memory warning; see above.
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # The sheer number of lookups makes this function horribly slow when
    # searching DataFrames, even with Polars. Converting to set lookups
    # in order to optimize type categorization for each ID.
    artist_ids = set(artist_data['ArtistID'])
    genre_ids = set(genre_data['GenreID'])
    album_ids = set(album_data['AlbumID'])
    
    # Fixed columns
    user_ids = []
    item_ids = []
    item_types = []
    ratings = []
    
    # Handling the UserID|Count format
    i = 0
    while i < len(lines):
        user_id, n = map(int, lines[i].strip().split('|'))
        i += 1
        for _ in range(int(n)):
            parts = lines[i].strip().split('\t')
            item_id = int(parts[0])
            rating = int(parts[1])
            
            # Determine the item type based on the presence of the ID in different sets
            if item_id in artist_ids:
                item_type = 'Artist'
            elif item_id in genre_ids:
                item_type = 'Genre'
            elif item_id in album_ids:
                item_type = 'Album'
            else:
                item_type = 'Track'
            
            user_ids.append(user_id)
            item_ids.append(item_id)
            item_types.append(item_type)
            ratings.append(rating)
            i += 1 
    
    # Final DataFrame construction
    df = pl.DataFrame({
        'UserID': user_ids,
        'ItemID': item_ids,
        'ItemType': item_types,
        'Rating': ratings
    })
    return df

# Used in other functions
artist_data = load_artist_data("artistData2.txt")
genre_data = load_genre_data("genreData2.txt")

# Read album data and write to csv
album_data = load_album_data("albumData2.txt")
album_data.write_csv("album_table.csv")

# Read track data and write to csv
track_data = load_track_data("trackData2.txt")
track_data.write_csv("track_table.csv")

# Read training data and write to csv
training_data = load_training_data("trainItem2.txt", artist_data, genre_data, album_data)
training_data.write_csv("training_table.csv")

# Read testing data and write to csv
test_data = load_test_data("testItem2.txt")
test_data.write_csv("testing_table.csv")