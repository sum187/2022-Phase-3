import * as React from 'react';
import { Box, Grid, Paper, Skeleton} from "@mui/material";

export default function GoTopagewithFollowingDetail({name,width}:any) {
  return (
    <Paper sx={{ backgroundColor: "#E0FFFF" }}>
      <Grid
        container
        direction="row"
        spacing={0}
        sx={{justifyContent: "center",}}>
        <Grid item>
          <Box sx={{ width: width}}>
          <h2>No result for {name}</h2>
          <h2>Search Help</h2>
          <ul>
            <li>Check your search for typos</li>
            <li>Use more generic search terms</li>
            <li>The movie you're searching for may be not on our site yet</li>
          </ul>
            <Skeleton />
            <Skeleton animation="wave" />
            <Skeleton animation={false} />
          </Box>
        </Grid>    
      </Grid>
    </Paper> 
  );
}