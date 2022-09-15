import * as React from 'react';
import {Grid, Paper,ImageList,ImageListItem,ImageListItemBar} from "@mui/material";
import IconButton from '@mui/material/IconButton';
import InfoIcon from '@mui/icons-material/Info';

function GoTopage({MovieInfo,col,width,height,page,values}:any) {
  /*link to noimage sign */
  const noImage = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/330px-No-Image-Placeholder.svg.png?20200912122019"

  return (
    <div> 
      <Paper sx={{ backgroundColor: "#191a1a" }}>
        <div
          className='App'
          id="movie-result"
          style={{
            maxWidth: "85%",
            margin: "0 auto",
            padding: "2.5vw 2.5vw 2.5vw 2.5vw",
          }}
        >
          <Paper sx={{ backgroundColor: "#030303" }}>
            <ImageList variant="standard" cols={col} gap={12}>
              {MovieInfo.Search.map((search:any,i:any) => (       
                <ImageListItem key={i}>
                  <img
                    src={`${validLink(search.Poster)}?w=248&fit=crop&auto=format`}
                    srcSet={`${validLink(search.Poster)}?w=248&fit=crop&auto=format&dpr=2 2x`}
                    alt={search.Title}
                    style={{ width: width,
                        height: height}}
                    loading="lazy"
                  />  
                  <ImageListItemBar 
                    title={search.Title} 
                  />
                </ImageListItem>
              ))}
            </ImageList>
          </Paper>
          <Grid
            container
            direction="row"
            spacing={0}
            sx={{
              justifyContent: "center",
            }}>
            <Paper sx={{ backgroundColor: "#a2a8a5" }}>
              <div
                id="movie-result"
                style={{
                  maxWidth: "100%",
                  margin: "0 auto",
                  //padding: "100px 10px 0px 10px",
                }}
              >
              </div>
            </Paper>
          </Grid>
        </div>
      </Paper>
    </div>
  );
  
  // if invalid get image that says 'no image'
  function validLink(value: any) {
    if ( value === "N/A") {
      return noImage;
    } else {
      return value;
    }
  }
}

export default GoTopage