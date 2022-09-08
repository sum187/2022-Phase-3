import React from 'react';

function displayImage(props) {
    const ingredientListItems = props.ingredients.map((ingredient, index) => {
        return (
            <li key={index} 
                className={ ingredient.prepared ? 'prepared' : '' }
                // TODO: Add onClick event
                onClick={ () => props.onClick(index) }
                
            >
                { ingredient.name }
            </li>
        );
    });

    return (
        <ul>
            { ingredientListItems }
        </ul>
    );
}

export default displayImage;