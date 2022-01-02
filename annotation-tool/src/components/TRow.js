import React from 'react'

import './TRow.css'


const TRow = ({count, row}) => {
   
    return (

        <tr>
            <td>{count++}</td>
            {
            row.map((td) => (
                <td >{td} </td>
            ))
            }
        </tr>
    )
}

export default TRow
