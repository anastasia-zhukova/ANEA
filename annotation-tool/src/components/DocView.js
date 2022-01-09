import React from 'react'
import { useState } from 'react/cjs/react.development';
import './DocView.css';
import arrowIcon from '../assets/arrow-white.svg'

import {FcCollapse} from 'react-icons/fc'







const DocView = ({text, id}) => {
    const [collapsed, setCollapsed] = useState(true);

    //console.log(texts);
    

    const returnComp = () => {

        if(!collapsed)
            return <>
                    <div className='docView-container'>
                        <div className="uncollapsed">
                            <img className='uncollapsedIcon' src={arrowIcon} alt="collapse icon" onClick={()=>(setCollapsed(!collapsed))}/>
                            <h2>{`Text ${id+1}`}</h2>
                        </div>
                        <div className='text-cont'>
                            {text}
                        </div>
            
                    </div>
                </>
        else
            return <>
                    <div className='collapsed'>
                        <img className='collapseIcon' src={arrowIcon} alt="collapse icon" onClick={()=>(setCollapsed(!collapsed))}/>
                        <h2>{`Text ${id+1}`}</h2>
    
                    </div>
                </>     
        
    }
    return (
        returnComp()
    )
}

export default DocView
