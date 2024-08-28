using Doji.AI.Depth;
using Doji.AI.Depth.Samples;
using Doji.AI.Diffusers;
using Unity.Sentis;
using UnityEngine;

namespace MyProject {
    public class WorldGeneration : MonoBehaviour {

        public string Prompt;

        private DiffusionPipeline _sd;
        private Midas _midas;
        DepthPointRenderer _renderer;

        private const int WIDTH = 512;
        private const int HEIGHT = 512;

        public RenderTexture SDResult;

        private void Start() {
            _midas = new Midas(ModelType.midas_v21_small_256);
            _sd = DiffusionPipeline.FromPretrained(DiffusionModel.SD_XL_TURBO);
            _renderer = gameObject.GetComponent<DepthPointRenderer>();
            _renderer.enabled = false;
            SDResult = new RenderTexture(WIDTH, HEIGHT, 0, RenderTextureFormat.ARGB32);
        }

        private void Update() {
            if (Input.GetKeyDown(KeyCode.Space)) {
                _renderer.enabled = true;
                var imageT = _sd.Generate(Prompt, numInferenceSteps: 1, guidanceScale: 1f);
                TextureConverter.RenderToTexture(imageT, SDResult);
                _midas.EstimateDepth(SDResult);
                var depth = _midas.Result;
                var extrema = _midas.GetMinMax();
                _renderer.MinPred = extrema.min;
                _renderer.MaxPred = extrema.max;
                _renderer.Depth = depth;
                _renderer.Source = SDResult;
            }
        }
    }
}
