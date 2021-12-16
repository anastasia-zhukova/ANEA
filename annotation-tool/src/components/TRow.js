import React from 'react'




const TRow = ({row}) => {
    return (

        <tr>
            {
            row.map((td) => (
                <td >{td} </td>
            ))
            }
        </tr>
    )
}

export default TRow
