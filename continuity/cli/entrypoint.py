import typer

app = typer.Typer()


@app.command()
def hello():
    print("Hello , Welcome to Continuity 😀.")


@app.command()
def make(file_path:str):
    # logic for compiling the model will be here .. 
    print("compiling and inferencing your model.")


if __name__ == "__main__":
    app()