// LandingPage.js
import { Link } from "react-router-dom";
import React from "react";
import "../css/Landing.css";
import background from "../images/black2.png";
import PageImg from "../images/6.jpg";
import trainer from "../images/4.jpg";
function LandingPage() {
  return (
    <div className="Container">
      <img alt="TrainerImage" className="TrainerImg" src={trainer} />
      <img alt="BackgroundImage" className="Background" src={background} />
      <img alt="LionImage" className="LionImg" src={PageImg} />
      <div className="navbar">
        <ul>
          <li>
            <Link to="/Homepage">LearnReflect</Link>
          </li>
          <li>
            <Link to="/ShopPage">Shop</Link>
          </li>
          <li>
            <Link to="/AIUpscalePage">LearnReflect AI</Link>
          </li>
          <li>
            <Link to="/Contact">Contact</Link>
          </li>
        </ul>
      </div>
    </div>
  );
}
export default LandingPage;
 