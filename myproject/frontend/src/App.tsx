import axios from "axios";
import { useState, useEffect } from 'react';
import * as React from 'react';
import './App.css';
import { Box, Button, Grid, Paper, Skeleton,ImageList,ImageListItem,ImageListItemBar,
  TextField} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import Pagination from '@mui/material/Pagination';
import Stack from '@mui/material/Stack';
import { Typography } from '@mui/material';
/* eslint-disable no-unused-expressions */

function App() {
  const [movieName, setMovieName] = useState<undefined | any>(undefined);
  const [name, setName] = useState<undefined | any>(undefined);
  const [MovieInfo, setMovieInfo] = useState<undefined | any>(undefined);
  const [page, setPage] = React.useState(1);
  const [searchResult, setSearchResult] = useState<undefined | any>(undefined);
  const handleChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
    setSearchResult([name,page]);
  };

  /*useEffect(() => {
   
    checkMovieName(name,page);
    console.log(name,page);
  
  });*/


  useEffect(() => {
   
    checkMovieName(name,page);
    console.log(name,page);
  
  },[searchResult]);
  /*setSearchResult([name,page])*/
  /*console.log(name,page)*/


  const Movie_BASE_URL = "https://www.omdbapi.com";
  const key = "8059c2e4";
  const noImage = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/330px-No-Image-Placeholder.svg.png?20200912122019"

  return (
    <div>
      <div className="search-field">
        <h1>Movie Search</h1 >
        <div style={{ display: "flex", justifyContent: "center" }}>
          <TextField
            id="search-bar"
            className="text"
            value={movieName}
            onChange={(prop) => {
              setMovieName(prop.target.value);
            }}
            label="Enter the moive name..."
            variant="outlined"
            placeholder="Search..."
            size="medium"
          />
          <Button
            onClick={() => {
              {movieName===undefined? (/* eslint-disable-line */
                setName(undefined)
              ) : (
                search('1'),
                setPage(1),
                setName(movieName)
              )}
            }}
          >
            <SearchIcon style={{ fill: "blue" }} />
            Search
          </Button>
        </div>
      </div>
      {MovieInfo === undefined? (
        <div></div>
      ) : MovieInfo.Response === "False" ? (
      <div
          id="movie-result"
          style={{
            maxWidth: "80%",
            margin: "0 auto",
            padding: "100px 10px 0px 10px",
          }}
        >
          <Paper sx={{ backgroundColor: "#E0FFFF" }}>
            <Grid
              container
              direction="row"
              spacing={5}
              sx={{
                justifyContent: "center",
              }}
          >
              <Grid item>
                <Box>
                  <h1>movie not found</h1>
                  <Skeleton width={300} height={300} />
                </Box>
              </Grid>    
            </Grid>
          </Paper> 
        </div> 
      ) : (
          <Paper sx={{ backgroundColor: "#191a1a" }}>
            <div
              id="movie-result"
              style={{
                maxWidth: "80%",
                margin: "0 auto",
                padding: "100px 10px 0px 10px",
              }}
            >
              <Paper sx={{ backgroundColor: "#030303" }}>
                <ImageList variant="standard" cols={5} gap={12}>
                  {MovieInfo.Search.map((search: any) => (       
                    <ImageListItem >
                      <img
                        src={`${validLink(search.Poster)}?w=248&fit=crop&auto=format`}
                        srcSet={`${validLink(search.Poster)}?w=248&fit=crop&auto=format&dpr=2 2x`}
                        alt={search.Title}
                        loading="lazy"
                      />  
                      <ImageListItemBar 
                        title={search.Title} 
                        subtitle={search.Director}
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
                    <Stack spacing={2}>
                      <Typography>Page: {page}</Typography>
                      <Pagination count={
                        Math.ceil(parseInt(MovieInfo.totalResults)/10)} 
                        variant="outlined"
                        shape="rounded"
                        page={page} 
                        onChange={handleChange} 
                        />
                    </Stack>
                  </div>
                </Paper>
              </Grid>
            </div>
          </Paper>
      )}   
    </div>
  );

  function search(page: any){
    axios
    .get(Movie_BASE_URL + "/?s=" + movieName + "&apikey=" + key + "&page=" +page).then((res) => {
      /*console.log(Math.ceil(10/3));*/
      setMovieInfo(res.data);
    });
  } 

  function validLink(value: any) {
    if ( value === "N/A") {
      return noImage;
    } else {
      return value;
    }
  }

  function checkMovieName(value: any,page:any){
    if (value === undefined) {
      return;
    } else {
      axios
      .get(Movie_BASE_URL + "/?s=" + value + "&apikey=" + key + "&type=movie&page=" +page)
      .then((res) => {
        setMovieInfo(res.data);
      })
    }}
}

export default App;
