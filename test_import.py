try:
    import generate_infinite_video
    print("Successfully imported generate_infinite_video")
except Exception as e:
    print(f"Import error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()