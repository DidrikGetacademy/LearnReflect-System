// LandingPage.js
import { Link } from 'react-router-dom';
import React from 'react';
import '../css/Landing.css'
import background from '../images/black2.png';
import PageImg from '../images/lion.jpg';
import trainer from '../images/4.jpg';
import black from '../images/black2.png';

function LandingPage() {
  return (
    <div className='banner'>
    <img alt="TrainerImage" className='TrainerImg' src={trainer}/>
    <img alt="BackgroundImage" className='Background' src={background}/>
    <img alt="LionImage" className='LionImg' src={PageImg}/>
    <div className='navbar'>
    <ul>
    <li><Link to='/Homepage'>LearnReflect</Link></li>
      <li><Link to='/ShopPage' >Shop</Link></li>
      <li><Link to='/AIUpscalePage'>LearnReflect AI</Link></li>
      <li><Link to='/Contact'>Contact</Link></li>
    </ul>
    </div>
    <div className='content'>
    <h1>LearnReflect Official</h1>
    <p>Learning Resources</p>
    <p>Improvement</p>
    <p>Inspiration</p>  
    <p>Guide</p>
    </div>
    <div className='banner2'>
    <img alt="LearnReflectImage" className='LearnReflectImg' src={black}/>
    </div>
    </div>
  );
}
export default LandingPage;
