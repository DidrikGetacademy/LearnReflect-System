import React from 'react';
import { Link } from 'react-router-dom';
function Bcomponent(){
return(
<div className='button-container'>
        <Link to='/Homepage'>
          <button>LearnReflect</button>
        </Link>
        <Link to='/LR'>
          <button>Official</button>
        </Link>
        <Link to='/Discipline'>
          <button>Discipline</button>
        </Link>
      </div>
)};

export default Bcomponent;