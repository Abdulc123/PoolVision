# These dimensions define the size of the output warped image (top-down view of the table).
WARPED_TABLE_W, WARPED_TABLE_H = 1000, 500  # Width and height of the table in pixels


# MAP each marker ID → its desired (x, y) in that top-down view
# This dictionary maps ArUco marker IDs to their corresponding positions in the warped (top-down) view.
# Marker IDs 0–3 are used as the corners of the table.
WARPED_DEST_PT = {
    0: [0,         0],          # Top-left corner of the table
    1: [WARPED_TABLE_W,   0],          # Top-right corner of the table
    2: [WARPED_TABLE_W,   WARPED_TABLE_H],    # Bottom-right corner of the table
    3: [0,         WARPED_TABLE_H],    # Bottom-left corner of the table
}

