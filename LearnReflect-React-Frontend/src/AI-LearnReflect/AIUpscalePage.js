import React from "react";
import { useNavigate } from "react-router-dom";
function AIUpscalePage() {
  const navigate = useNavigate();
  return (
    <div>
      <h1>AI Upscale Page</h1>
      <div className="AIupscale-Container">
        <div>
          <img alt="AudioEnchancer" />
          <button onClick={() => navigate('/Chatbot')}>Audio Enchancer</button>
          <p>
            This is a description of the Audio Enchancer button.
          </p>
        </div>

        <div>
          <img alt="VideoEnchancer" />
          <button onClick={() => navigate('/')}>Video Enchancer</button>
          <p>
            This is a description of the Video Enchancer button.
          </p>
        </div>

        <div>
          <img alt="Chatbot" />
          <button onClick={() => navigate('/')}>AI Chatbot</button>
          <p>
            This is a description of the LearnReflect Chatbot button.
          </p>
        </div>
      </div>
    </div>
  );
}
export default AIUpscalePage;
