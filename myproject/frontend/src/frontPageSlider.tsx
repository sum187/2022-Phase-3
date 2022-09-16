import SimpleImageSlider from "react-simple-image-slider";
import {Paper} from "@mui/material";
import * as React from 'react';

export default function GoTopagewithFollowingDetail({width,height}:any) {

    // url and key 
    const Movie_BASE_URL = "https://www.omdbapi.com";
    //const key = "938b78ef";
    const key = "8059c2e4";
    //const key = "52776e8"

    const images = [
        { url: "https://m.media-amazon.com/images/M/MV5BMzVlMmY2NTctODgwOC00NDMzLWEzMWYtM2RiYmIyNTNhMTI0XkEyXkFqcGdeQXVyNTAzNzgwNTg@._V1_SX300.jpg" },
        { url: "https://upload.wikimedia.org/wikipedia/en/d/d4/Uncharted_Official_Poster.jpg" },
        { url: "https://upload.wikimedia.org/wikipedia/en/2/2f/Morbius_%28film%29_poster.jpg" },
        { url: "https://upload.wikimedia.org/wikipedia/en/3/34/Fantastic_Beasts-_The_Secrets_of_Dumbledore.png?20220503232805" },
        { url: "https://upload.wikimedia.org/wikipedia/en/8/88/Thor_Love_and_Thunder_poster.jpeg?20220524020524" },
        { url: "https://upload.wikimedia.org/wikipedia/en/1/13/Top_Gun_Maverick_Poster.jpg" },
      ];

    return (
        <div> 
      <Paper sx={{ backgroundColor: "#191a1a" }}>
        <div
          className='App'
          id="movie-result"
          style={{
            maxWidth: "94%",
            margin: "0 auto",
            padding: "2vw 2vw 2vw 2vw",
          }}
        >
          <Paper sx={{ backgroundColor: "#030303" }}>
            <div style={{ display: "flex", justifyContent: "center" }}>
                <SimpleImageSlider
                    width={width}
                    height={height}
                    images={images}
                    showBullets={true}
                    showNavs={true}
                />
            </div>
          </Paper>
        </div>
      </Paper>
    </div>  
    );
  }