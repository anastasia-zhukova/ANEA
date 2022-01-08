import React from 'react'

import Tdata from './Tdata';
import './TRow.css'


const TRow = ({count, row, delTdata, editData}) => {

    var tDataId = 0;
    return (
        
        <tr>
            <td>{count++}</td>
            {
                row.map((td) => (
                    <Tdata editData={editData}  delTdata={delTdata} key={tDataId++} /*rowId= {count}*/ id={tDataId} data = {td}/>
                    
                ))
            }
        </tr> 
    )
}

export default TRow
