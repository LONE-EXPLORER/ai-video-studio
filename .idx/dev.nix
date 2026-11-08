# Configuration for Google Project IDX / Google Cloud Workspaces
{ pkgs, ... }: {
  channel = "stable-24.05";
  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv
    pkgs.ffmpeg
  ];
  env = {};
  idx = {
    extensions = [
      "ms-python.python"
    ];
    workspace = {
      onCreate = {
        setup = "python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt";
      };
      onStart = {
        run-studio = "source .venv/bin/activate && streamlit run app/main.py --server.port 8501 --server.headless true";
      };
    };
    previews = {
      enable = true;
      previews = {
        web = {
          command = ["source", ".venv/bin/activate", "&&", "streamlit", "run", "app/main.py", "--server.port", "$PORT", "--server.headless", "true"];
          manager = "web";
        };
      };
    };
  };
}
