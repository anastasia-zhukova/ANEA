
import './AppBody.css'

import React from 'react'

const AppBody = () => {
    return (
        <div className='body-container'>
            <h1> To start please choose a document or a dataset</h1>
            <div className='inputs-container'>
                <input type='file' multiple='false' id='doc-input' accept='.pdf, .doc, .docx'/>
                <label htmlFor='doc-input'>
                    Select document
                </label>

                <input type='file' multiple='true' id='data-input' accept='.json' />
                
                <label htmlFor='data-input'>
                    Select dataset
                </label>


                
            </div>
        </div>
    )
}

export default AppBody
